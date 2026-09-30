// Adapted from mizorewww/laya-mlx. See NOTICE for attribution and pinned sources.
import MLX
import MLXNN

final class LayaEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokens: Embedding
    @ModuleInfo var norm: LayerNorm

    init(_ c: LayaEncoderConfiguration) {
        _tokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        norm = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray { norm(tokens(ids)) }
}

final class LayaEncoderAttention: Module {
    let heads: Int
    let headDimension: Int
    let rope: RoPE
    @ModuleInfo var Wqkv: Linear
    @ModuleInfo var Wo: Linear

    init(_ c: LayaEncoderConfiguration, index: Int) {
        heads = c.numAttentionHeads
        headDimension = c.hiddenSize / heads
        rope = RoPE(dimensions: headDimension, traditional: false, base: c.ropeBases[index])
        Wqkv = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: c.attentionBias)
        Wo = Linear(c.hiddenSize, c.hiddenSize, bias: c.attentionBias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(1)
        let qkv = Wqkv(x).reshaped(b, length, 3, heads, headDimension)
        let q = rope(qkv[0..., 0..., 0].transposed(0, 2, 1, 3), offset: 0)
        let k = rope(qkv[0..., 0..., 1].transposed(0, 2, 1, 3), offset: 0)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDimension).squareRoot(), mask: mask)
        return Wo(output.transposed(0, 2, 1, 3).reshaped(b, length, -1))
    }
}

final class LayaEncoderMLP: Module {
    @ModuleInfo var Wi: Linear
    @ModuleInfo var Wo: Linear

    init(_ c: LayaEncoderConfiguration) {
        Wi = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: c.mlpBias)
        Wo = Linear(c.intermediateSize, c.hiddenSize, bias: c.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(Wi(x), parts: 2, axis: -1)
        return Wo(gelu(parts[0]) * parts[1])
    }
}

final class LayaEncoderLayer: Module {
    let isGlobal: Bool
    @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm?
    @ModuleInfo var attn: LayaEncoderAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo var mlp: LayaEncoderMLP

    init(_ c: LayaEncoderConfiguration, index: Int) {
        isGlobal = c.layerTypes[index] == "full_attention"
        _attentionNorm.wrappedValue = index == 0 ? nil
            : LayerNorm(dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
        attn = LayaEncoderAttention(c, index: index)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
        mlp = LayaEncoderMLP(c)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attn(attentionNorm?(x) ?? x, mask: mask)
        return h + mlp(mlpNorm(h))
    }
}

func layaAttentionMasks(_ mask: MLXArray, window: Int) -> (full: MLXArray, local: MLXArray) {
    let valid = mask.asType(.bool)
    let full = valid[0..., .newAxis, .newAxis, 0...]
    let positions = MLXArray(0..<mask.dim(1))
    let nearby = abs(positions[0..., .newAxis] - positions[.newAxis, 0...]) .<= (window / 2)
    // Padded queries still see valid keys, avoiding all-masked softmax rows.
    let local = logicalAnd(
        logicalOr(nearby[.newAxis, .newAxis], logicalNot(valid[0..., .newAxis, 0..., .newAxis])), full)
    return (full, local)
}

final class LayaEncoder: Module {
    let window: Int
    @ModuleInfo var embeddings: LayaEmbeddings
    @ModuleInfo var layers: [LayaEncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ c: LayaEncoderConfiguration) {
        window = c.localAttention
        embeddings = LayaEmbeddings(c)
        layers = (0..<c.numHiddenLayers).map { LayaEncoderLayer(c, index: $0) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        var h = embeddings(ids)
        let masks = layaAttentionMasks(mask, window: window)
        for layer in layers { h = layer(h, mask: layer.isGlobal ? masks.full : masks.local) }
        return finalNorm(h)
    }
}

final class LayaHeadAttention: Module {
    let heads: Int
    let headDimension: Int
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    init(dimensions: Int) {
        heads = max(1, dimensions / 64)
        headDimension = dimensions / heads
        _input.wrappedValue = Linear(dimensions, 3 * dimensions)
        _output.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(1)
        let qkv = input(x).reshaped(b, length, 3, heads, headDimension)
        let q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)
        let h = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDimension).squareRoot(), mask: mask)
        return output(h.transposed(0, 2, 1, 3).reshaped(b, length, -1))
    }
}

final class LayaHeadLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LayaHeadAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(dimensions: Int) {
        _attention.wrappedValue = LayaHeadAttention(dimensions: dimensions)
        norm1 = LayerNorm(dimensions: dimensions)
        norm2 = LayerNorm(dimensions: dimensions)
        linear1 = Linear(dimensions, 4 * dimensions)
        linear2 = Linear(4 * dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x), mask: mask)
        return h + linear2(relu(linear1(norm2(h))))
    }
}

final class LayaHead: Module {
    @ModuleInfo var layers: [LayaHeadLayer]

    init(dimensions: Int, count: Int) {
        layers = (0..<count).map { _ in LayaHeadLayer(dimensions: dimensions) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h, mask: mask) }
        return h
    }
}

final class LayaNetwork: Module {
    @ModuleInfo var encoder: LayaEncoder
    @ModuleInfo var head: LayaHead
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo var scorer: Sequential
    @ModuleInfo(key: "act_head") var actionHead: Sequential
    @ParameterInfo var temperature = MLXArray.ones([3])

    init(encoder c: LayaEncoderConfiguration, agent: LayaAgentConfiguration) {
        encoder = LayaEncoder(c)
        head = LayaHead(dimensions: c.hiddenSize, count: agent.headLayers)
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: c.hiddenSize)
        scorer = Sequential {
            LayerNorm(dimensions: c.hiddenSize)
            Linear(c.hiddenSize, c.hiddenSize)
            GELU()
            Linear(c.hiddenSize, 1)
        }
        _actionHead.wrappedValue = Sequential {
            Linear(c.hiddenSize + 4, 256)
            GELU()
            Linear(256, agent.actionCount)
        }
    }

    func callAsFunction(ids: MLXArray, mask: MLXArray, positions: MLXArray,
                        markerMask: MLXArray, types: MLXArray) -> (MLXArray, MLXArray) {
        var h = encoder(ids, mask: mask)
        h = h + typeEmbedding(types)[0..., .newAxis, 0...]
        h = head(h, mask: mask[0..., .newAxis, .newAxis, 0...].asType(.bool))
        let markers = h[MLXArray(0..<h.dim(0))[0..., .newAxis], positions]
        let logits = which(markerMask, scorer(markers).squeezed(axis: -1).asType(.float32), -1e4)
        let p = softmax(logits, axis: -1)
        let k = maximum(markerMask.sum(axis: -1), 2).asType(.float32)
        let entropy = -(p * log(maximum(p, 1e-9))).sum(axis: -1) / log(k)
        let top = sorted(p, axis: -1)
        let features = stacked([top[0..., -1], top[0..., -1] - top[0..., -2], entropy, k / 255], axis: -1)
        let pooled = concatenated([h[0..., 0].asType(.float32), features], axis: -1)
        let action = actionHead(pooled.asType(h.dtype))
        return (logits, action.asType(.float32))
    }
}
