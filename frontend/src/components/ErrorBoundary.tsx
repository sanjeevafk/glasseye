import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Catches render/runtime errors so a single failing component never blanks the app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    console.error("ErrorBoundary caught:", error, info.componentStack ?? "");
  }

  render() {
    if (this.state.error) {
      return (
        <div className="error-boundary" role="alert">
          <p className="section-kicker">DASHBOARD ERROR</p>
          <h2>Something went wrong rendering this section.</h2>
          <p className="muted">{this.state.error.message}</p>
          <button type="button" onClick={() => this.setState({ error: null })}>
            TRY AGAIN
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
