package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"IA_RAG/repo"
	"net/http"
)

// RAGService orchestrates chunking, embeddings and repository operations
// and LLM non-stream generation if needed.
type RAGService struct {
	repo           repo.DocumentRepository
	httpClient     *http.Client
	ollamaURL      string
	embeddingModel string
	llmModel       string
	chunkSize      int
	chunkOverlap   int
}

type ollamaEmbedResp struct {
	Embedding []float32 `json:"embedding"`
}

func NewRAGService(r repo.DocumentRepository, httpClient *http.Client, ollamaURL, embeddingModel, llmModel string, chunkSize, chunkOverlap int) *RAGService {
	return &RAGService{
		repo:           r,
		httpClient:     httpClient,
		ollamaURL:      ollamaURL,
		embeddingModel: embeddingModel,
		llmModel:       llmModel,
		chunkSize:      chunkSize,
		chunkOverlap:   chunkOverlap,
	}
}

func (s *RAGService) ChunkText(text string) []string {
	words := strings.Fields(text)
	var chunks []string
	if s.chunkSize <= 0 {
		return []string{text}
	}
	step := s.chunkSize - s.chunkOverlap
	if step <= 0 {
		step = s.chunkSize
	}
	for i := 0; i < len(words); i += step {
		end := i + s.chunkSize
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

func (s *RAGService) GenerateEmbedding(text string) ([]float32, error) {
	reqBody := map[string]any{
		"model":  s.embeddingModel,
		"prompt": text,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	resp, err := s.httpClient.Post(s.ollamaURL+"/api/embeddings", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error calling ollama embeddings: %w", err)
	}
	defer resp.Body.Close()
	var result ollamaEmbedResp
	dec := json.NewDecoder(resp.Body)
	if resp.StatusCode != http.StatusOK {
		var raw map[string]any
		_ = dec.Decode(&raw)
		return nil, fmt.Errorf("ollama embeddings status %d: %v", resp.StatusCode, raw)
	}
	if err := dec.Decode(&result); err != nil {
		return nil, fmt.Errorf("error parsing embeddings JSON: %w", err)
	}
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("ollama embeddings returned empty vector")
	}
	return result.Embedding, nil
}

// IndexDocument chunks the content, embeds each chunk and stores it via repository
func (s *RAGService) IndexDocument(ctx context.Context, content, source string) error {
	chunks := s.ChunkText(content)
	for i, ch := range chunks {
		emb, err := s.GenerateEmbedding(ch)
		if err != nil {
			return fmt.Errorf("embedding chunk %d: %w", i, err)
		}
		if err := s.repo.InsertChunk(ctx, ch, source, emb); err != nil {
			return fmt.Errorf("storing chunk %d: %w", i, err)
		}
	}
	return nil
}

// SearchSimilarContents embeds the question and retrieves similar chunks' contents only
func (s *RAGService) SearchSimilarContents(ctx context.Context, question string, topK int) ([]string, error) {
	emb, err := s.GenerateEmbedding(question)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	docs, err := s.repo.SearchSimilar(ctx, emb, topK)
	if err != nil {
		return nil, err
	}
	contents := make([]string, 0, len(docs))
	for _, d := range docs {
		contents = append(contents, d.Content)
	}
	return contents, nil
}

func (s *RAGService) HTTPClient() *http.Client { return s.httpClient }

func (s *RAGService) LLMModel() string { return s.llmModel }

func (s *RAGService) OllamaURL() string { return s.ollamaURL }
