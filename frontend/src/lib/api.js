const API_BASE_URL =
  (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

function toErrorMessage(status, payload) {
  if (payload && typeof payload.detail === 'string') {
    return payload.detail
  }

  if (payload && typeof payload.error === 'string') {
    return payload.error
  }

  return `Request failed with status ${status}.`
}

export async function predictSentencePair(body, options = {}) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal: options.signal,
  })

  let payload = null

  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    throw new Error(toErrorMessage(response.status, payload))
  }

  return payload
}
