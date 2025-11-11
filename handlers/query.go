package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// NewQueryHandler builds an SSE handler that:
// - uses searchFn to fetch relevant chunk contents for a question (topK configurable via query param 'k', default applied)
// - calls Ollama with stream=true and forwards tokens as Server-Sent Events
func NewQueryHandler(
	searchFn func(ctx context.Context, question string, topK int) ([]string, error),
	llmModel string,
	ollamaURL string,
	httpClient *http.Client,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		question := strings.TrimSpace(r.URL.Query().Get("q"))
		if question == "" {
			http.Error(w, "missing parameter 'q'", http.StatusBadRequest)
			return
		}

		topK := 50

		docs, err := searchFn(r.Context(), question, topK)
		if err != nil {
			http.Error(w, fmt.Sprintf("error looking for context: %v", err), http.StatusInternalServerError)
			return
		}

		var contextStr strings.Builder
		contextStr.WriteString("Relevant context:\n\n")
		for i, content := range docs {
			contextStr.WriteString(fmt.Sprintf("[%d] %s\n\n", i+1, content))
		}
		prompt := fmt.Sprintf("%s\nPregunta: %s\nInstrucciones: Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está en el contexto, indica que no tienes suficiente información.\nRespuesta:", contextStr.String(), question)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		reqBody := map[string]interface{}{
			"model":  llmModel,
			"prompt": prompt,
			"stream": true,
		}
		jsonData, _ := json.Marshal(reqBody)
		ollamaResp, err := httpClient.Post(ollamaURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			http.Error(w, fmt.Sprintf("error calling ollama: %v", err), http.StatusBadGateway)
			return
		}
		defer ollamaResp.Body.Close()

		dec := json.NewDecoder(ollamaResp.Body)
		for {
			var chunk struct {
				Response string `json:"response"`
				Done     bool   `json:"done"`
			}
			if err := dec.Decode(&chunk); err != nil {
				if err == io.EOF {
					break
				}
				fmt.Fprintf(w, "event: error\n")
				fmt.Fprintf(w, "data: %s\n\n", strings.ReplaceAll(err.Error(), "\n", " "))
				flusher.Flush()
				break
			}

			if chunk.Response != "" {
				fmt.Fprintf(w, "data: %s\n\n", strings.ReplaceAll(chunk.Response, "\n", "\\n"))
				flusher.Flush()
			}
			if chunk.Done {
				fmt.Fprintf(w, "event: done\n")
				fmt.Fprintf(w, "data: done\n\n")
				flusher.Flush()
				break
			}
		}
	}
}
