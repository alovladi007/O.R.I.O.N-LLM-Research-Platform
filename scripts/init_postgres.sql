-- NANO-OS PostgreSQL Initialization Script
-- This script is run when the PostgreSQL container starts for the first time
-- It creates the database, enables extensions, and sets up basic configuration

-- Create database (if running as postgres user)
-- Note: In docker-compose, the database is created automatically via POSTGRES_DB
-- This is here for documentation and manual setup

-- Connect to the NANO-OS database
\c orion_db;

-- Enable required PostgreSQL extensions
-- pgvector: Vector similarity search for ML/embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- UUID generation functions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pg_trgm: Trigram similarity for text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- btree_gin: GIN indexes for better query performance
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- PostGIS: Spatial data support (future use for crystal structures)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- Display installed extensions
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm', 'btree_gin')
ORDER BY extname;

-- Create schema for vector embeddings if needed
-- CREATE SCHEMA IF NOT EXISTS ml;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE orion_db TO orion;
GRANT ALL PRIVILEGES ON SCHEMA public TO orion;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO orion;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO orion;

-- Configure PostgreSQL for optimal performance
-- These settings are conservative; adjust based on your hardware

-- Shared buffers (25% of RAM recommended for dedicated DB server)
-- This will be overridden by postgresql.conf or docker-compose environment

-- Work memory for sorting/hashing (per operation)
ALTER SYSTEM SET work_mem = '16MB';

-- Maintenance work memory (for VACUUM, CREATE INDEX, etc.)
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Effective cache size (hint to planner about OS cache)
ALTER SYSTEM SET effective_cache_size = '2GB';

-- Random page cost (lower for SSD)
ALTER SYSTEM SET random_page_cost = 1.1;

-- Enable parallel query execution
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Logging configuration for development
ALTER SYSTEM SET log_statement = 'mod';  -- Log all DDL/DML statements
ALTER SYSTEM SET log_duration = on;       -- Log statement durations
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log slow queries (>1s)

-- Reload configuration
SELECT pg_reload_conf();

-- Display current configuration
SHOW shared_buffers;
SHOW work_mem;
SHOW maintenance_work_mem;
SHOW effective_cache_size;

-- Verify database and user
SELECT current_database(), current_user;

-- Display message
DO $$
BEGIN
  RAISE NOTICE '========================================';
  RAISE NOTICE 'NANO-OS PostgreSQL initialization complete!';
  RAISE NOTICE '========================================';
  RAISE NOTICE 'Database: orion_db';
  RAISE NOTICE 'Extensions enabled: vector, uuid-ossp, pg_trgm, btree_gin';
  RAISE NOTICE 'Next steps:';
  RAISE NOTICE '  1. Run Alembic migrations: alembic upgrade head';
  RAISE NOTICE '  2. Start the API server';
  RAISE NOTICE '  3. Start the Celery worker';
  RAISE NOTICE '========================================';
END $$;
