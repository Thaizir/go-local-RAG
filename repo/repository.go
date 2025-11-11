package repo

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	github_com_pgv "github.com/pgvector/pgvector-go"
)

// Document represents a stored chunk
type Document struct {
	ID      int
	Content string
	Source  string
	Vector  github_com_pgv.Vector
}

// DocumentRepository abstracts DB operations for RAG
type DocumentRepository interface {
	Init(ctx context.Context) error
	InsertChunk(ctx context.Context, content, source string, embedding []float32) error
	SearchSimilar(ctx context.Context, queryEmbedding []float32, topK int) ([]Document, error)
	Close(ctx context.Context) error
}

// PostgresRepository implements DocumentRepository using pgx and pgvector
type PostgresRepository struct {
	conn *pgx.Conn
}

func NewPostgresRepository(ctx context.Context, dbURL string) (*PostgresRepository, error) {
	conn, err := pgx.Connect(ctx, dbURL)
	if err != nil {
		return nil, fmt.Errorf("error connecting to postgres: %w", err)
	}
	return &PostgresRepository{conn: conn}, nil
}

func (p *PostgresRepository) Close(ctx context.Context) error {
	return p.conn.Close(ctx)
}

func (p *PostgresRepository) Init(ctx context.Context) error {
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
	for _, q := range queries {
		if _, err := p.conn.Exec(ctx, q); err != nil {
			return fmt.Errorf("error executing init query: %w", err)
		}
	}
	return nil
}

func (p *PostgresRepository) InsertChunk(ctx context.Context, content, source string, embedding []float32) error {
	_, err := p.conn.Exec(ctx,
		"INSERT INTO documents (content, source, embedding) VALUES ($1, $2, $3)",
		content, source, github_com_pgv.NewVector(embedding),
	)
	if err != nil {
		return fmt.Errorf("error inserting chunk: %w", err)
	}
	return nil
}

func (p *PostgresRepository) SearchSimilar(ctx context.Context, queryEmbedding []float32, topK int) ([]Document, error) {
	rows, err := p.conn.Query(ctx,
		`SELECT id, content, source, embedding FROM documents ORDER BY embedding <=> $1 LIMIT $2`,
		github_com_pgv.NewVector(queryEmbedding), topK,
	)
	if err != nil {
		return nil, fmt.Errorf("error performing vector search: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.Content, &d.Source, &d.Vector); err != nil {
			return nil, err
		}
		docs = append(docs, d)
	}
	return docs, nil
}
