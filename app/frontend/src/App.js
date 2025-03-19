import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [history, setHistory] = useState([]);

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!query.trim()) {
            setError('Please enter a query');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const result = await axios.post('/api/process', { query });
            setResponse(result.data.result);

            // Add to history
            setHistory([
                {
                    id: Date.now(),
                    query,
                    response: result.data.result
                },
                ...history
            ]);

            // Clear query
            setQuery('');
        } catch (err) {
            setError(err.response?.data?.error || 'An error occurred. Please try again.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // Example queries
    const exampleQueries = [
        "Run a seizure detector on the last 24 hours of data",
        "Analyze sleep stages from the last 8 hours",
        "Calculate spectral power in the alpha band for the last hour",
        "What's the brain synchrony in delta and theta bands over the last 2 hours?",
    ];

    const handleExampleClick = (example) => {
        setQuery(example);
    };

    return (
        <div className="app">
            <header className="header">
                <h1>AskEEG</h1>
                <p>Natural Language EEG Analysis</p>
            </header>

            <main className="main">
                <section className="query-section">
                    <form onSubmit={handleSubmit}>
                        <div className="input-container">
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Ask about EEG data (e.g., Run a seizure detector on the last 24 hours)"
                                disabled={loading}
                            />
                            <button type="submit" disabled={loading}>
                                {loading ? 'Processing...' : 'Ask'}
                            </button>
                        </div>
                        {error && <div className="error">{error}</div>}
                    </form>

                    <div className="examples">
                        <h3>Example queries:</h3>
                        <div className="example-buttons">
                            {exampleQueries.map((example, index) => (
                                <button
                                    key={index}
                                    onClick={() => handleExampleClick(example)}
                                    className="example-button"
                                >
                                    {example}
                                </button>
                            ))}
                        </div>
                    </div>
                </section>

                {response && (
                    <section className="response-section">
                        <h2>Analysis Results</h2>
                        <div className="response-content">
                            <pre>{response}</pre>
                        </div>
                    </section>
                )}

                {history.length > 0 && (
                    <section className="history-section">
                        <h2>Query History</h2>
                        <div className="history-list">
                            {history.map((item) => (
                                <div key={item.id} className="history-item">
                                    <div className="history-query">
                                        <strong>Query:</strong> {item.query}
                                    </div>
                                    <div className="history-response">
                                        <strong>Response:</strong>
                                        <pre>{item.response}</pre>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}
            </main>

            <footer className="footer">
                <p>© {new Date().getFullYear()} AskEEG - Natural Language EEG Analysis</p>
            </footer>
        </div>
    );
}

export default App; 