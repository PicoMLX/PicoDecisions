import Foundation
import Testing
import PicoDecisions
@testable import PicoDecisionsMLX

@Test func rejectsInvalidTemperatures() {
    let json = #"{"encoder":"test","head_layers":2,"temperature":[1,0,1]}"#
    #expect(throws: DecisionError.self) {
        _ = try JSONDecoder().decode(LayaAgentConfiguration.self, from: Data(json.utf8))
    }
}

@Test func calibratesByQuestionAndOptionCount() throws {
    let json = #"{"encoder":"test","head_layers":2,"temperature":[2,3,4],"temperature_by_options":{"choice:3-5":0.5}}"#
    let config = try JSONDecoder().decode(LayaAgentConfiguration.self, from: Data(json.utf8))
    #expect(config.temperature(type: 0, options: 3) == 0.5)
    #expect(config.temperature(type: 0, options: 2) == 2)
    #expect(config.temperature(type: 2, options: 2) == 4)
}

@Test func stableSoftmaxAndInvalidOutput() throws {
    let p = try layaSoftmax([10_000, 10_000])
    #expect(p == [0.5, 0.5])
    #expect(throws: DecisionError.self) { try layaSoftmax([.nan, 0]) }
    #expect(throws: DecisionError.self) { try layaSoftmax([]) }
}
