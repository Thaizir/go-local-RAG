package handlers

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// NewUploadHandler returns a handler that accepts multipart form with optional text and/or .txt file
// indexFn should persist content and its source into the vector DB.
func NewUploadHandler(indexFn func(ctx context.Context, content, source string) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		if err := r.ParseMultipartForm(10 << 20); err != nil { // 10MB
			http.Error(w, fmt.Sprintf("error parsing form: %v", err), http.StatusBadRequest)
			return
		}

		text := r.FormValue("text")
		var source string
		var content string

		file, header, err := r.FormFile("file")
		if err == nil {
			defer file.Close()
			if !strings.HasSuffix(strings.ToLower(header.Filename), ".txt") {
				http.Error(w, "solo se aceptan archivos .txt", http.StatusBadRequest)
				return
			}
			b, err := io.ReadAll(file)
			if err != nil {
				http.Error(w, fmt.Sprintf("error leyendo archivo: %v", err), http.StatusBadRequest)
				return
			}
			content = string(b)
			source = header.Filename
		}

		if content == "" {
			content = text
			source = "user_text"
		}

		if strings.TrimSpace(content) == "" {
			http.Error(w, "no se proporcionó texto ni archivo", http.StatusBadRequest)
			return
		}

		log.Printf("Indexing new content from %s (len=%d)", source, len(content))
		if err := indexFn(r.Context(), content, source); err != nil {
			http.Error(w, fmt.Sprintf("error indexando documento: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}
}
