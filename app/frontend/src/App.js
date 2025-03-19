import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [history, setHistory] = useState([]);
    const [tools, setTools] = useState([]);
    const [toolsLoading, setToolsLoading] = useState(true);
    const [toolsError, setToolsError] = useState('');

    // Fetch available tools on component mount
    useEffect(() => {
        const fetchTools = async () => {
            try {
                const result = await axios.get('/api/tools');
                setTools(result.data.tools);
                setToolsLoading(false);
            } catch (err) {
                setToolsError('Failed to load available tools');
                console.error(err);
                setToolsLoading(false);

                // Fallback to hardcoded tools in case API doesn't work yet
                setTools([
                    {
                        name: "synchrony_analyzer",
                        description: "Analyze brain synchrony between EEG channels using the Kuramoto order parameter. Useful for measuring how synchronized different brain regions are.",
                        parameters: {
                            "data": "The EEG data to analyze as a numpy array"
                        }
                    },
                    {
                        name: "alpha_delta_ratio",
                        description: "Calculate the ratio of alpha to delta power. This is useful for awareness analysis.",
                        parameters: {
                            "data": "The EEG data to analyze as a numpy array",
                            "fs": "The sampling frequency (Hz) of the data"
                        }
                    },
                    {
                        name: "spike_detector",
                        description: "Detect epileptiform spikes in EEG data. This is useful for seizure monitoring and epilepsy diagnosis.",
                        parameters: {
                            "data": "The EEG data to analyze as a numpy array",
                            "fs": "The sampling frequency (Hz) of the data"
                        }
                    }
                ]);
            }
        };

        fetchTools();
    }, []);

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
        "Can you detect seizure activity in my brain recordings?",
        "I'd like to analyze my sleep patterns over the last 8 hours",
        "What's my alpha to delta ratio?",
        "Is there high synchrony between my frontal and parietal brain regions?"
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
                                placeholder="Ask about EEG data (e.g., Detect seizure activity in my brain recordings)"
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

                <section className="tools-section">
                    <h2>Available Analysis Tools</h2>
                    <p className="tools-description">
                        AskEEG contains several specialized tools for analyzing your brain activity data.
                        You can reference these tools in your queries or simply ask questions about your EEG data.
                    </p>
                    {toolsLoading ? (
                        <p>Loading available tools...</p>
                    ) : toolsError ? (
                        <p className="error">{toolsError}</p>
                    ) : (
                        <div className="tools-list">
                            {tools.map((tool, index) => (
                                <div key={index} className="tool-card">
                                    <h3>{tool.name}</h3>
                                    <p>{tool.description}</p>
                                    <div className="tool-parameters">
                                        <h4>Parameters:</h4>
                                        <ul>
                                            {Object.entries(tool.parameters).map(([key, value], i) => (
                                                <li key={i}>
                                                    <strong>{key}:</strong> {value}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </main>
        </div>
    );
}

export default App; 