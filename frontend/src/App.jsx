import { startTransition, useMemo, useState } from 'react'
import './App.css'
import { predictSentencePair } from './lib/api.js'
import { datasetResults } from './lib/resultsData.js'

const PARAPHRASE_MODELS = [
  'Siamese-LSTM',
  'Siamese-GRU',
  'BERT',
  'RoBERTa',
  'DistilBERT',
]

const SIMILARITY_MODELS = [
  'Siamese-LSTM',
  'Siamese-GRU',
  'BERT',
  'RoBERTa',
  'DistilBERT',
  'SBERT',
]

const EXAMPLE_INPUT = {
  sentence1: 'How can I learn natural language processing quickly?',
  sentence2: 'What is the fastest way to study NLP effectively?',
}

function classifyTone(label) {
  const normalised = String(label ?? '').toLowerCase()

  if (normalised.includes('not')) {
    return 'negative'
  }

  if (normalised.includes('para') || normalised.includes('similar')) {
    return 'positive'
  }

  return 'neutral'
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'N/A'
  }

  const numeric = Number(value)
  if (numeric <= 1) {
    return `${(numeric * 100).toFixed(1)}%`
  }

  return `${numeric.toFixed(1)}%`
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'N/A'
  }

  return Number(value).toFixed(3)
}

function getClassificationEntry(results, modelName) {
  return results?.paraphrase_detection?.[modelName] ?? null
}

function getSimilarityEntry(results, modelName) {
  return results?.semantic_similarity?.[modelName] ?? null
}

function App() {
  const [sentence1, setSentence1] = useState(EXAMPLE_INPUT.sentence1)
  const [sentence2, setSentence2] = useState(EXAMPLE_INPUT.sentence2)
  const [validationError, setValidationError] = useState('')
  const [requestError, setRequestError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [results, setResults] = useState(null)

  const hasResults = Boolean(results)

  const metaCards = useMemo(
    () => [
      {
        title: 'Paraphrase Detection',
        value: '5 models',
        detail: 'Binary classification across Siamese and transformer systems.',
      },
      {
        title: 'Semantic Similarity',
        value: '6 models',
        detail: 'Continuous similarity scoring, including SBERT zero-shot output.',
      },
      {
        title: 'Backend Contract',
        value: 'POST /predict',
        detail: 'Frontend expects JSON results keyed by model name.',
      },
    ],
    [],
  )

  async function handleSubmit(event) {
    event.preventDefault()

    const trimmedSentence1 = sentence1.trim()
    const trimmedSentence2 = sentence2.trim()

    if (!trimmedSentence1 || !trimmedSentence2) {
      setValidationError('Both sentence fields are required before running inference.')
      setRequestError('')
      return
    }

    setValidationError('')
    setRequestError('')
    setIsSubmitting(true)

    try {
      const response = await predictSentencePair({
        sentence1: trimmedSentence1,
        sentence2: trimmedSentence2,
      })

      startTransition(() => {
        setResults(response)
      })
    } catch (error) {
      setRequestError(error.message || 'The prediction request failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  function handleFillExample() {
    setSentence1(EXAMPLE_INPUT.sentence1)
    setSentence2(EXAMPLE_INPUT.sentence2)
    setValidationError('')
    setRequestError('')
  }

  function handleClear() {
    setSentence1('')
    setSentence2('')
    setValidationError('')
    setRequestError('')
    setResults(null)
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">Semantic Comparison Studio</p>
        <div className="hero-copy">
          <div>
            <h1>Test sentence pairs across every model in the project.</h1>
            <p className="lede">
              Enter two sentences and compare paraphrase classification alongside
              semantic similarity scores from the Siamese baselines, fine-tuned
              transformers, and SBERT.
            </p>
          </div>

          <div className="meta-grid">
            {metaCards.map((card) => (
              <article className="meta-card" key={card.title}>
                <p className="meta-title">{card.title}</p>
                <strong>{card.value}</strong>
                <span>{card.detail}</span>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="input-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <div>
              <p className="section-label">Input</p>
              <h2>Sentence Pair</h2>
            </div>
            <div className="button-row">
              <button className="ghost-button" onClick={handleFillExample} type="button">
                Use Example
              </button>
              <button className="ghost-button" onClick={handleClear} type="button">
                Clear
              </button>
            </div>
          </div>

          <label className="field">
            <span>Sentence 1</span>
            <textarea
              value={sentence1}
              onChange={(event) => setSentence1(event.target.value)}
              placeholder="Enter the first sentence..."
              rows={5}
            />
          </label>

          <label className="field">
            <span>Sentence 2</span>
            <textarea
              value={sentence2}
              onChange={(event) => setSentence2(event.target.value)}
              placeholder="Enter the second sentence..."
              rows={5}
            />
          </label>

          <div className="submit-row">
            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Running models...' : 'Compare Sentences'}
            </button>
            <p className="endpoint-note">
              API target: <code>POST /predict</code>
            </p>
          </div>

          {validationError ? <p className="message error">{validationError}</p> : null}
          {requestError ? <p className="message error">{requestError}</p> : null}
          {isSubmitting ? (
            <p className="message loading">
              Sending the pair to the backend and waiting for model outputs.
            </p>
          ) : null}
        </form>

        <section className="results-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Output</p>
              <h2>Model Results</h2>
            </div>
            <span className={`status-pill ${hasResults ? 'ready' : 'idle'}`}>
              {hasResults ? 'Response loaded' : 'Waiting for inference'}
            </span>
          </div>

          {!hasResults ? (
            <div className="empty-state">
              <p>
                Results will appear here after the backend returns paraphrase and
                similarity predictions for the submitted pair.
              </p>
            </div>
          ) : (
            <div className="result-sections">
              <section className="result-card">
                <div className="result-card-header">
                  <div>
                    <p className="section-label">Task A</p>
                    <h3>Paraphrase Detection</h3>
                  </div>
                  <p className="result-description">
                    Binary classification and optional confidence from the models
                    trained for paraphrase detection.
                  </p>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {PARAPHRASE_MODELS.map((modelName) => {
                        const entry = getClassificationEntry(results, modelName)
                        const label = entry?.label ?? 'Unavailable'
                        const confidence = entry?.confidence

                        return (
                          <tr key={modelName}>
                            <td>{modelName}</td>
                            <td>
                              <span className={`tag ${classifyTone(label)}`}>{label}</span>
                            </td>
                            <td>{formatConfidence(confidence)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="result-card">
                <div className="result-card-header">
                  <div>
                    <p className="section-label">Task B</p>
                    <h3>Semantic Similarity</h3>
                  </div>
                  <p className="result-description">
                    Continuous similarity scores returned by the STS-capable
                    models currently in the project.
                  </p>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Scale</th>
                      </tr>
                    </thead>
                    <tbody>
                      {SIMILARITY_MODELS.map((modelName) => {
                        const entry = getSimilarityEntry(results, modelName)

                        return (
                          <tr key={modelName}>
                            <td>{modelName}</td>
                            <td>{formatScore(entry?.score)}</td>
                            <td>{entry?.scale ?? 'N/A'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          )}
        </section>
      </section>

      <section className="training-results-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Evaluation</p>
            <h2>Training & Benchmark Results</h2>
          </div>
          <p className="result-description">
            Comprehensive evaluation metrics and architecture complexity analysis across MRPC, QQP, and STS-B datasets.
          </p>
        </div>

        <div className="dataset-results-grid">
          {/* MRPC Results */}
          <article className="result-card dataset-card">
            <div className="result-card-header">
              <h3>MRPC (Paraphrase)</h3>
              <p className="endpoint-note">Binary Classification</p>
            </div>
            
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>Time (s)</th>
                    <th>Params</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(datasetResults.mrpc).map(([model, metrics]) => (
                    <tr key={model}>
                      <td>{model}</td>
                      <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                      <td>{(metrics.f1 * 100).toFixed(1)}</td>
                      <td>{metrics.time.toFixed(3)}</td>
                      <td>{(metrics.total_params / 1000000).toFixed(1)}M</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="plot-gallery">
              <img src="/plots/mrpc/accuracy.png" alt="MRPC Accuracy" loading="lazy" />
              <img src="/plots/mrpc/f1.png" alt="MRPC F1 Score" loading="lazy" />
              <img src="/plots/mrpc/complexity_tradeoff.png" alt="MRPC Complexity Tradeoff" loading="lazy" className="full-width-plot" />
            </div>
          </article>

          {/* QQP Results */}
          <article className="result-card dataset-card">
            <div className="result-card-header">
              <h3>QQP (Question Pairs)</h3>
              <p className="endpoint-note">Binary Classification</p>
            </div>
            
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>Time (s)</th>
                    <th>Params</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(datasetResults.qqp).map(([model, metrics]) => (
                    <tr key={model}>
                      <td>{model}</td>
                      <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                      <td>{(metrics.f1 * 100).toFixed(1)}</td>
                      <td>{metrics.time.toFixed(3)}</td>
                      <td>{(metrics.total_params / 1000000).toFixed(1)}M</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="plot-gallery">
              <img src="/plots/qqp/accuracy.png" alt="QQP Accuracy" loading="lazy" />
              <img src="/plots/qqp/f1.png" alt="QQP F1 Score" loading="lazy" />
              <img src="/plots/qqp/complexity_tradeoff.png" alt="QQP Complexity Tradeoff" loading="lazy" className="full-width-plot" />
            </div>
          </article>

          {/* STS Results */}
          <article className="result-card dataset-card">
            <div className="result-card-header">
              <h3>STS-B (Semantic Similarity)</h3>
              <p className="endpoint-note">Regression</p>
            </div>
            
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Pearson</th>
                    <th>Spearman</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(datasetResults.sts).map(([model, metrics]) => (
                    <tr key={model}>
                      <td>{model}</td>
                      <td>{metrics.pearson.toFixed(3)}</td>
                      <td>{metrics.spearman.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="plot-gallery">
              <img src="/plots/sts/sts_correlations.png" alt="STS Correlations" loading="lazy" className="full-width-plot" />
            </div>
          </article>
        </div>
      </section>
    </main>
  )
}

export default App
