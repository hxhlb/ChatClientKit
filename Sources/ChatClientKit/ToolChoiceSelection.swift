import Foundation

extension ChatRequestBody {
    func selectedTools() throws -> [Tool] {
        let tools = tools ?? []
        switch toolChoice {
        case .auto, nil:
            return tools
        case .required:
            guard !tools.isEmpty else {
                throw toolChoiceError(
                    String(localized: "A required tool choice needs at least one tool.")
                )
            }
            return tools
        case let .function(selectedName):
            guard let selected = tools.first(where: { $0.functionName == selectedName }) else {
                let key: String.LocalizationValue = "The selected tool '\(selectedName)' is not available."
                throw toolChoiceError(String(localized: key))
            }
            return [selected]
        }
    }

    var toolChoiceDirective: String? {
        switch toolChoice {
        case .required:
            "Call one of the available tools before answering the user."
        case let .function(name):
            "Call the \(name) tool. Do not answer the user directly."
        case .auto, nil:
            nil
        }
    }

    private func toolChoiceError(_ description: String) -> NSError {
        NSError(
            domain: "ChatClientKit.ToolChoice",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }
}

private extension ChatRequestBody.Tool {
    var functionName: String {
        switch self {
        case let .function(name, _, _, _):
            name
        }
    }
}
