import SwiftUI

/// Placeholder root view for the scanning interface.
/// Will be replaced with the full scanning UI in task 11.1.
struct ContentView: View {
    var body: some View {
        VStack {
            Text("Area Target Scanner")
                .font(.largeTitle)
            Text("Ready to scan")
                .foregroundColor(.secondary)
        }
        .padding()
    }
}
