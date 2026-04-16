import { startTransition, useMemo, useState } from 'react'
import './App.css'
import { predictSentencePair } from './lib/api.js'
import { datasetResults } from './lib/resultsData.js'

const DATASET_TABS = [
  { key: 'mrpc', label: 'MRPC', subtitle: 'Paraphrase Detection' },
  { key: 'qqp', label: 'QQP', subtitle: 'Question Pairs' },
  { key: 'stsb', label: 'STS-B', subtitle: 'Semantic Similarity' },
]

const CLASSIFICATION_MODELS = [
  'Siamese-LSTM',
  'Siamese-GRU',
  'BERT',
  'RoBERTa',
  'DistilBERT',
]

const SIMILARITY_MODELS_BASE = [
  'Siamese-LSTM',
  'Siamese-GRU',
  'BERT',
  'RoBERTa',
  'DistilBERT',
]

const SIMILARITY_MODELS_TUNED = [
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
  if (normalised.includes('not')) return 'negative'
  if (normalised.includes('para') || normalised.includes('similar')) return 'positive'
  return 'neutral'
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'
  const numeric = Number(value)
  if (numeric <= 1) return `${(numeric * 100).toFixed(1)}%`
  return `${numeric.toFixed(1)}%`
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'
  return Number(value).toFixed(3)
}

function App() {
  const [sentence1, setSentence1] = useState(EXAMPLE_INPUT.sentence1)
  const [sentence2, setSentence2] = useState(EXAMPLE_INPUT.sentence2)
  const [validationError, setValidationError] = useState('')
  const [requestError, setRequestError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [results, setResults] = useState(null)
  const [activeDataset, setActiveDataset] = useState('mrpc')
  const [activeBenchmarkDataset, setActiveBenchmarkDataset] = useState('mrpc')

  const hasResults = Boolean(results)

  const metaCards = useMemo(
    () => [
      {
        title: 'Datasets',
        value: '3',
        detail: 'MRPC, QQP, and STS-B benchmark datasets.',
      },
      {
        title: 'Model Variants',
        value: 'Base + Tuned',
        detail: 'Pre-trained (no fine-tuning) vs. fine-tuned models.',
      },
      {
        title: 'Total Predictions',
        value: '32',
        detail: '3 datasets × 2 variants × 5 models + SBERT.',
      },
    ],
    [],
  )

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmedS1 = sentence1.trim()
    const trimmedS2 = sentence2.trim()
    if (!trimmedS1 || !trimmedS2) {
      setValidationError('Both sentence fields are required.')
      setRequestError('')
      return
    }
    setValidationError('')
    setRequestError('')
    setIsSubmitting(true)
    try {
      const response = await predictSentencePair({ sentence1: trimmedS1, sentence2: trimmedS2 })
      startTransition(() => setResults(response))
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

  // Get live prediction data for a variant
  function getVariantPredictions(datasetKey, variant) {
    return results?.[datasetKey]?.[variant] ?? {}
  }

  function getDatasetTask(datasetKey) {
    return results?.[datasetKey]?.task ?? (datasetKey === 'stsb' ? 'semantic_similarity' : 'paraphrase_detection')
  }

  // Render a classification table for a variant
  function renderClassificationTable(predictions, variant) {
    return (
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
            {CLASSIFICATION_MODELS.map((modelName) => {
              const entry = predictions[modelName]
              const label = entry?.label ?? 'Unavailable'
              const confidence = entry?.confidence
              return (
                <tr key={modelName}>
                  <td>{modelName}</td>
                  <td><span className={`tag ${classifyTone(label)}`}>{label}</span></td>
                  <td>{formatConfidence(confidence)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }

  // Render a similarity table for a variant
  function renderSimilarityTable(predictions, variant) {
    const models = variant === 'tuned' ? SIMILARITY_MODELS_TUNED : SIMILARITY_MODELS_BASE
    return (
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
            {models.map((modelName) => {
              const entry = predictions[modelName]
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
    )
  }

  // Render benchmark results for a dataset
  function renderBenchmarkDataset(datasetKey) {
    const dsData = datasetResults[datasetKey]
    if (!dsData) return null
    const isClassification = dsData.task === 'paraphrase_detection'

    return (
      <div className="benchmark-variants">
        {['base', 'tuned'].map((variant) => {
          const variantData = dsData[variant]
          if (!variantData) return null
          return (
            <div className="benchmark-variant-col" key={variant}>
              <h4 className="variant-heading">
                <span className={`variant-badge ${variant}`}>{variant === 'base' ? 'Base (Pre-trained)' : 'Tuned (Fine-tuned)'}</span>
              </h4>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      {isClassification ? (
                        <>
                          <th>Accuracy</th>
                          <th>F1</th>
                          <th>Time (s)</th>
                        </>
                      ) : (
                        <>
                          <th>Pearson</th>
                          <th>Spearman</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(variantData).map(([model, metrics]) => (
                      <tr key={model}>
                        <td>{model}</td>
                        {isClassification ? (
                          <>
                            <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                            <td>{(metrics.f1 * 100).toFixed(1)}</td>
                            <td>{metrics.time.toFixed(3)}</td>
                          </>
                        ) : (
                          <>
                            <td>{metrics.pearson.toFixed(3)}</td>
                            <td>{metrics.spearman.toFixed(3)}</td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">Semantic Comparison Studio</p>
        <div className="hero-copy">
          <div>
            <h1>Test sentence pairs across every model in the project.</h1>
            <p className="lede">
              Enter two sentences and compare predictions from base (pre-trained)
              and tuned (fine-tuned) models across MRPC, QQP, and STS-B datasets.
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
              <button className="ghost-button" onClick={handleFillExample} type="button">Use Example</button>
              <button className="ghost-button" onClick={handleClear} type="button">Clear</button>
            </div>
          </div>

          <label className="field">
            <span>Sentence 1</span>
            <textarea value={sentence1} onChange={(e) => setSentence1(e.target.value)} placeholder="Enter the first sentence..." rows={5} />
          </label>

          <label className="field">
            <span>Sentence 2</span>
            <textarea value={sentence2} onChange={(e) => setSentence2(e.target.value)} placeholder="Enter the second sentence..." rows={5} />
          </label>

          <div className="submit-row">
            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Running models...' : 'Compare Sentences'}
            </button>
            <p className="endpoint-note">API target: <code>POST /predict</code></p>
          </div>

          {validationError ? <p className="message error">{validationError}</p> : null}
          {requestError ? <p className="message error">{requestError}</p> : null}
          {isSubmitting ? <p className="message loading">Sending the pair to the backend and waiting for model outputs.</p> : null}
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
              <p>Results will appear here after the backend returns predictions from all datasets and variants.</p>
            </div>
          ) : (
            <>
              {/* Dataset tabs */}
              <div className="dataset-tabs">
                {DATASET_TABS.map((tab) => (
                  <button
                    key={tab.key}
                    className={`dataset-tab ${activeDataset === tab.key ? 'active' : ''}`}
                    onClick={() => setActiveDataset(tab.key)}
                    type="button"
                  >
                    <strong>{tab.label}</strong>
                    <span>{tab.subtitle}</span>
                  </button>
                ))}
              </div>

              {/* Base vs Tuned comparison */}
              <div className="variant-comparison">
                {['base', 'tuned'].map((variant) => {
                  const predictions = getVariantPredictions(activeDataset, variant)
                  const task = getDatasetTask(activeDataset)
                  const isClassification = task === 'paraphrase_detection'
                  return (
                    <section className="variant-column" key={variant}>
                      <div className="variant-header">
                        <span className={`variant-badge ${variant}`}>
                          {variant === 'base' ? 'Base (Pre-trained)' : 'Tuned (Fine-tuned)'}
                        </span>
                      </div>
                      {isClassification
                        ? renderClassificationTable(predictions, variant)
                        : renderSimilarityTable(predictions, variant)}
                    </section>
                  )
                })}
              </div>
            </>
          )}
        </section>
      </section>

      {/* Training & Benchmark Results */}
      <section className="training-results-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Evaluation</p>
            <h2>Training & Benchmark Results</h2>
          </div>
          <p className="result-description">
            Base (pre-trained, no fine-tuning) vs Tuned (fine-tuned) performance across all datasets.
          </p>
        </div>

        <div className="dataset-tabs benchmark-tabs">
          {DATASET_TABS.map((tab) => (
            <button
              key={tab.key}
              className={`dataset-tab ${activeBenchmarkDataset === tab.key ? 'active' : ''}`}
              onClick={() => setActiveBenchmarkDataset(tab.key)}
              type="button"
            >
              <strong>{tab.label}</strong>
              <span>{tab.subtitle}</span>
            </button>
          ))}
        </div>

        {renderBenchmarkDataset(activeBenchmarkDataset)}
      </section>
    </main>
  )
}

export default App
