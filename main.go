package main

import (
	"IA_RAG/handlers"
	"IA_RAG/repo"
	"IA_RAG/service"
	"context"
	"log"
	"net/http"
	"time"
)

const (
	ollamaURL      = "http://localhost:11434"
	embeddingModel = "nomic-embed-text"
	llmModel       = "llama3.2"

	dbURL        = "postgres://raguser:ragpass@localhost:5432/ragdb?sslmode=disable"
	chunkSize    = 500
	chunkOverlap = 100
)

func main() {
	ctx := context.Background()

	// HTTP client shared by the service
	httpClient := &http.Client{Timeout: 60 * time.Second}

	// Repository (DB)
	dbRepo, err := repo.NewPostgresRepository(ctx, dbURL)
	if err != nil {
		log.Fatal(err)
	}
	defer dbRepo.Close(ctx)
	if err := dbRepo.Init(ctx); err != nil {
		log.Fatal(err)
	}
	log.Println("✓ database initialized")

	svc := service.NewRAGService(dbRepo, httpClient, ollamaURL, embeddingModel, llmModel, chunkSize, chunkOverlap)
	mux := http.NewServeMux()

	fileServer := http.FileServer(http.Dir("web"))
	mux.Handle("/", fileServer)

	// Healthcheck
	mux.HandleFunc("/api/health", handlers.NewHealthHandler())

	// Upload endpoint: accepts text or .txt file
	mux.HandleFunc("/api/upload", handlers.NewUploadHandler(svc.IndexDocument))

	// Query endpoint with SSE streaming, using service search and direct LLM streaming in handler
	mux.HandleFunc("/api/query", handlers.NewQueryHandler(
		svc.SearchSimilarContents,
		svc.LLMModel(),
		svc.OllamaURL(),
		svc.HTTPClient(),
	))

	addr := ":8080"
	log.Printf("Server running in %s — open http://localhost%s/", addr, addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
