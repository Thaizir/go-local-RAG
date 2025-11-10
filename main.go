package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"
)

const (
	ollamaURL      = "http://localhost:11434"
	embeddingModel = "nomic-embed-text"
	llmModel       = "llama3.2"

	dbURL        = "postgres://raguser:ragpass@localhost:5432/ragdb?sslmode=disable"
	chunkSize    = 500
	chunkOverlap = 100
)

type Document struct {
	ID      int
	Content string
	Source  string
	Vector  pgvector.Vector
}

// Main RAG client
type RAGClient struct {
	db   *pgx.Conn
	http *http.Client
}

// Ollama Response for embeddings
type OllamaEmdebResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Response for generation
type OllamaGenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func NewRagClient(ctx context.Context) (*RAGClient, error) {
	conn, err := pgx.Connect(ctx, dbURL)
	if err != nil {
		return nil, fmt.Errorf("error connectig to postgres db, %v", err)
	}

	return &RAGClient{
		db: conn,
		http: &http.Client{
			Timeout: 60 * time.Second,
		},
	}, nil
}

func (r *RAGClient) Close() {
	r.db.Close(context.Background())
}

// DATA stuff
func (r *RAGClient) InitDB(ctx context.Context) error {
	queries := []string{
		"CREATE EXTENSION IF NOT EXISTS vector",
		`CREATE TABLE IF NOT EXISTS documents (
			id SERIAL PRIMARY KEY,
			content TEXT NOT NULL,
			source TEXT NOT NULL,
			embedding vector(768)
		)`,
		"CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
	}

	for _, query := range queries {
		if _, err := r.db.Exec(ctx, query); err != nil {
			return fmt.Errorf("error executing query: %w", err)
		}
	}

	log.Println("✓ database initialized")
	return nil
}

func (r *RAGClient) ChunkText(text string) []string {
	words := strings.Fields(text)
	var chunks []string

	for i := 0; i < len(words); i += chunkSize - chunkOverlap {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)

		if end == len(words) {
			break
		}
	}
	return chunks
}

func (r *RAGClient) GenerateEmbedding(text string) ([]float32, error) {
	reqBody := map[string]interface{}{
		"model":  embeddingModel,
		"prompt": text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := r.http.Post(
		ollamaURL+"/api/embeddings",
		"application/json",
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, fmt.Errorf("error calling ollama embeddings: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading ollama embeddings response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embeddings status %d: %s", resp.StatusCode, string(body))
	}
	var result OllamaEmdebResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("error parsing embeddings JSON: %w. body: %s", err, string(body))
	}
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("ollama embeddings returned empty vector")
	}

	return result.Embedding, nil
}

func (r *RAGClient) IndexDocument(ctx context.Context, content, source string) error {
	chunks := r.ChunkText(content)
	log.Printf("document divided in %d chunks", len(chunks))

	for i, chunk := range chunks {
		embedding, err := r.GenerateEmbedding(chunk)
		if err != nil {
			return fmt.Errorf("error generating embedding for chunk %d: %w", i, err)
		}

		_, err = r.db.Exec(ctx,
			"INSERT INTO documents (content, source, embedding) VALUES ($1, $2, $3)",
			chunk, source, pgvector.NewVector(embedding),
		)

		if err != nil {
			return fmt.Errorf("error inserting chunck %d: %w", i, err)
		}
		log.Printf("indexed chunck %d/%d", i+1, len(chunks))
	}
	return nil
}

func (r *RAGClient) SearchSimilarChunks(ctx context.Context, query string, topK int) ([]Document, error) {

	queryEmbedding, err := r.GenerateEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("error generating embedding in query: %w", err)
	}

	rows, err := r.db.Query(ctx, `SELECT id, content, source, embedding FROM documents ORDER BY embedding <=> $1 LIMIT $2`, pgvector.NewVector(queryEmbedding), topK)
	if err != nil {
		return nil, fmt.Errorf("error in vectorial search: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var doc Document
		if err := rows.Scan(&doc.ID, &doc.Content, &doc.Source, &doc.Vector); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

func (r *RAGClient) GenerateAnswer(query string, context []Document) (string, error) {
	var contextStr strings.Builder
	contextStr.WriteString("Relevant context:\n\n")
	for i, doc := range context {
		contextStr.WriteString(fmt.Sprintf("[%d] %s\n\n", i+1, doc.Content))
	}

	prompt := fmt.Sprintf(`%s
Pregunta: %s
Instrucciones: Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está en el contexto, indica que no tienes suficiente información.
Respuesta:`, contextStr.String(), query)

	reqBody := map[string]interface{}{
		"model":  llmModel,
		"prompt": prompt,
		"stream": false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := r.http.Post(
		ollamaURL+"/api/generate",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return "", fmt.Errorf("error calling ollama: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading ollama response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var result OllamaGenerateResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("error parsing ollama JSON: %w. body: %s", err, string(body))
	}
	return result.Response, nil
}

func (r *RAGClient) Query(ctx context.Context, question string, topK int) (string, error) {
	log.Printf("Looking for relevant chunks for: '%s'", question)

	docs, err := r.SearchSimilarChunks(ctx, question, topK)
	if err != nil {
		return "", err
	}
	log.Printf("found %d relevant chunks", len(docs))

	log.Printf("Generating response with the LLM...")
	answer, err := r.GenerateAnswer(question, docs)
	if err != nil {
		return "", err
	}

	return answer, nil

}

func main() {

	ctx := context.Background()

	rag, err := NewRagClient(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer rag.Close()

	if err := rag.InitDB(ctx); err != nil {
		log.Fatal(err)
	}

	sampleDoc := `
	Go es un lenguaje de programación de código abierto creado por Google. 
	Fue diseñado por Thaizir El Troudi en 2010.
	Go es conocido por su simplicidad, eficiencia y excelente soporte para concurrencia.
	El lenguaje utiliza goroutines para manejar operaciones concurrentes de manera eficiente.
	Go se compila a código nativo, lo que lo hace muy rápido.array_to_halfvec
	Es ampliamente utilizado para desarrollo de microservicios, herramientas CLI y sistemas distribuidos.
	`

	log.Println("\n📚 indexing sample doct...")
	if err := rag.IndexDocument(ctx, sampleDoc, "documento_go.txt"); err != nil {
		log.Fatal(err)
	}

	log.Printf("\n querying...")
	question := "Quienes crearon el lenguaje Go?"
	answer, err := rag.Query(ctx, question, 100)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\n" + strings.Repeat("=", 60) + "\n")
	fmt.Printf("QUESTION: %s\n", question)
	fmt.Printf(strings.Repeat("=", 60) + "\n")
	fmt.Printf("ANSWER:\n%s\n", answer)
	fmt.Printf(strings.Repeat("=", 60) + "\n")
}
