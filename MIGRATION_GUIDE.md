# NANO-OS Database Migration Guide

This guide explains how to set up and use Alembic database migrations for the NANO-OS platform.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Running Migrations](#running-migrations)
- [Creating New Migrations](#creating-new-migrations)
- [Common Operations](#common-operations)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

NANO-OS uses Alembic for database schema version control and migrations. The migration system:

- Tracks all database schema changes in version-controlled Python files
- Supports both upgrade and downgrade operations
- Works with async SQLAlchemy and PostgreSQL with pgvector extension
- Provides automatic migration generation based on model changes

## Prerequisites

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `alembic>=1.12.0` - Migration tool
- `sqlalchemy>=2.0.0` - ORM framework
- `asyncpg>=0.29.0` - Async PostgreSQL driver
- `pgvector>=0.2.0` - Vector similarity search extension
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter

### 2. PostgreSQL Setup

Ensure PostgreSQL 12+ is installed with the pgvector extension:

```bash
# Install pgvector extension in PostgreSQL
sudo apt-get install postgresql-12-pgvector  # Ubuntu/Debian
# OR
brew install pgvector  # macOS with Homebrew

# Connect to your database
psql -U orion -d orion_db

# Enable extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure Database Connection

Set the `DATABASE_URL` environment variable:

```bash
# In your .env file or environment
export DATABASE_URL="postgresql+asyncpg://orion:your_password@localhost:5432/orion_db"
```

Or update `alembic.ini` with your database URL.

## Initial Setup

### 1. Verify Configuration

Check that Alembic can find your configuration:

```bash
cd /home/user/O.R.I.O.N-LLM-Research-Platform
alembic current
```

Expected output (for a fresh database):
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
```

### 2. Apply Initial Migration

Run the initial migration to create all tables:

```bash
alembic upgrade head
```

This creates the following tables:
- `users` - User accounts and authentication
- `materials` - Material metadata
- `structures` - Atomic structure data
- `workflow_templates` - Simulation workflow definitions
- `simulation_jobs` - Submitted simulation jobs
- `simulation_results` - Simulation outputs
- `vector_embeddings` - ML embeddings for similarity search
- `structure_similarities` - Precomputed structure similarities

### 3. Verify Migration

Check the current database version:

```bash
alembic current
```

Expected output:
```
001_initial_schema (head)
```

## Running Migrations

### Upgrade to Latest Version

```bash
alembic upgrade head
```

### Upgrade One Step

```bash
alembic upgrade +1
```

### Downgrade One Step

```bash
alembic downgrade -1
```

### Downgrade to Specific Version

```bash
alembic downgrade 001_initial_schema
```

### View Migration History

```bash
alembic history --verbose
```

## Creating New Migrations

### Auto-Generate Migration from Model Changes

When you modify models in `src/api/models/`, generate a migration:

```bash
# 1. Make your model changes in src/api/models/
# 2. Auto-generate migration
alembic revision --autogenerate -m "add email_verified column to users"

# 3. Review the generated migration file in alembic/versions/
# 4. Edit if necessary
# 5. Apply the migration
alembic upgrade head
```

### Create Empty Migration Template

For custom migrations (data migrations, complex changes):

```bash
alembic revision -m "migrate old data format"
# Edit the generated file and add your upgrade/downgrade logic
alembic upgrade head
```

## Common Operations

### Check Current Database Version

```bash
alembic current
```

### View Pending Migrations

```bash
alembic show head
```

### Stamp Database Without Running Migrations

If you manually created tables and want to mark them as migrated:

```bash
alembic stamp head
```

### Show SQL Without Running Migrations

Preview the SQL that would be executed:

```bash
alembic upgrade head --sql
```

### Generate SQL Script for Offline Migration

```bash
alembic upgrade head --sql > migration.sql
```

## Troubleshooting

### Error: "Target database is not up to date"

**Solution**: Apply pending migrations
```bash
alembic upgrade head
```

### Error: "Can't locate revision identified by..."

**Solution**: Database version table is out of sync. Stamp the current version:
```bash
alembic stamp head
```

### Error: "pgvector extension not found"

**Solution**: Install and enable pgvector
```bash
# Install pgvector in PostgreSQL
sudo apt-get install postgresql-12-pgvector

# Enable in database
psql -U orion -d orion_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Error: "No such file or directory: alembic.ini"

**Solution**: Run alembic commands from the project root directory:
```bash
cd /home/user/O.R.I.O.N-LLM-Research-Platform
alembic upgrade head
```

### Error: "relation already exists"

**Solution**: Either:
1. Drop existing tables and re-run migrations, OR
2. Stamp the database if tables were manually created:
```bash
alembic stamp head
```

### Connection Errors

**Solution**: Verify DATABASE_URL and database is running:
```bash
# Test connection
psql $DATABASE_URL -c "SELECT version();"

# Check PostgreSQL is running
sudo systemctl status postgresql
```

## Best Practices

### 1. Always Review Auto-Generated Migrations

Alembic's autogenerate is helpful but not perfect. Always review:
- Column type changes
- Index creations/deletions
- Foreign key constraints
- Enum changes

### 2. Test Migrations Locally First

```bash
# Test upgrade
alembic upgrade head

# Test downgrade
alembic downgrade -1

# Test upgrade again
alembic upgrade head
```

### 3. Backup Before Production Migrations

```bash
# Backup PostgreSQL database
pg_dump -U orion orion_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

### 4. One Logical Change Per Migration

Keep migrations focused and atomic:
- ✅ Good: "add index on materials.formula"
- ❌ Bad: "add indexes and modify columns and add tables"

### 5. Handle Data Migrations Carefully

For migrations that modify data:

```python
def upgrade():
    # 1. Add new column (nullable)
    op.add_column('materials', sa.Column('new_field', sa.String(), nullable=True))

    # 2. Migrate data
    connection = op.get_bind()
    connection.execute(
        "UPDATE materials SET new_field = old_field WHERE old_field IS NOT NULL"
    )

    # 3. Make not-nullable if needed
    op.alter_column('materials', 'new_field', nullable=False)

    # 4. Drop old column
    op.drop_column('materials', 'old_field')
```

### 6. Document Complex Migrations

Add detailed docstrings to migration files:

```python
"""Add email verification workflow

This migration:
1. Adds email_verified column to users table
2. Adds verification_token column
3. Creates index on verification_token
4. Sets email_verified=True for existing users

Revision ID: abc123
"""
```

### 7. Use Batch Operations for Large Tables

For large tables, use batch operations to avoid locks:

```python
with op.batch_alter_table('materials', schema=None) as batch_op:
    batch_op.add_column(sa.Column('new_field', sa.String()))
    batch_op.create_index('ix_materials_new_field', ['new_field'])
```

## Production Deployment Checklist

Before running migrations in production:

- [ ] Backup database
- [ ] Test migration on staging environment
- [ ] Review all SQL that will be executed (`alembic upgrade head --sql`)
- [ ] Plan for downtime if needed (for large table alterations)
- [ ] Have rollback plan ready
- [ ] Monitor application logs during and after migration
- [ ] Verify data integrity after migration

## Example Workflows

### Adding a New Table

```bash
# 1. Create model in src/api/models/
# 2. Import model in src/api/models/__init__.py
# 3. Generate migration
alembic revision --autogenerate -m "add experiment_results table"
# 4. Review and apply
alembic upgrade head
```

### Adding a Column

```bash
# 1. Add column to model
# 2. Generate migration
alembic revision --autogenerate -m "add temperature column to structures"
# 3. Review migration
cat alembic/versions/*_add_temperature*.py
# 4. Apply migration
alembic upgrade head
```

### Renaming a Column (with data preservation)

```bash
# 1. Create manual migration
alembic revision -m "rename old_name to new_name in materials"

# 2. Edit migration file:
# def upgrade():
#     op.alter_column('materials', 'old_name', new_column_name='new_name')
# def downgrade():
#     op.alter_column('materials', 'new_name', new_column_name='old_name')

# 3. Apply migration
alembic upgrade head
```

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## Support

For migration issues:
1. Check this guide's troubleshooting section
2. Review `alembic/README` for quick reference
3. Check Alembic logs for detailed error messages
4. Consult the NANO-OS development team
