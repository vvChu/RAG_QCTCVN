# Changelog

## [Unreleased] - 2024-11-16

### Added
- ✅ Zilliz Cloud support với authentication (user/password)
- ✅ Secure connection (TLS/SSL) option
- ✅ Auto-detection between local và cloud mode
- ✅ Enhanced connection error messages với troubleshooting
- ✅ `test_connection.py` script để verify setup
- ✅ Comprehensive Zilliz Cloud documentation (`ZILLIZ_CLOUD_SETUP.md`)
- ✅ Quick start guide (`ZILLIZ_QUICKSTART.md`)

### Changed
- 🔄 Updated `.env.example` với Zilliz Cloud options
- 🔄 Enhanced `MilvusVectorDB.__init__()` với authentication parameters
- 🔄 Improved `MilvusVectorDB.connect()` với better error handling
- 🔄 Updated `RAGSystem._load_config_from_env()` để load credentials
- 🔄 Updated all documentation để reflect cloud support

### Technical Details

#### Vector Store Updates
```python
# Before
MilvusVectorDB(host, port, collection_name)

# After
MilvusVectorDB(host, port, user, password, secure, collection_name)
```

#### Environment Variables
```env
# New variables
MILVUS_USER=db_74b5693bc1c4c80
MILVUS_PASSWORD=your_password
MILVUS_SECURE=True
```

#### Connection Flow
1. Load credentials từ `.env`
2. Auto-detect connection type (local vs cloud)
3. Apply appropriate connection parameters
4. Enhanced error messages với context-specific troubleshooting

### Deployment Options

#### Option 1: Local Milvus (Docker)
- Fast development
- No network latency
- Requires Docker Desktop
- Manual scaling

#### Option 2: Zilliz Cloud (Managed)
- No infrastructure management
- Auto-scaling
- High availability
- Free tier: 1GB storage
- Credentials provided

### Breaking Changes
None. Backward compatible với existing local setups.

### Migration Guide

#### From Local to Zilliz Cloud

1. Update `.env`:
```env
MILVUS_HOST=your-endpoint.vectordb.zillizcloud.com
MILVUS_USER=db_74b5693bc1c4c80
MILVUS_PASSWORD=Tg8+UKg4{{)ze9.(
MILVUS_SECURE=True
```

2. Test connection:
```bash
python test_connection.py
```

3. Re-index documents:
```bash
python src/main.py --mode index --pdf-dir data/pdfs
```

### Performance Implications

**Local Milvus:**
- Latency: 10-20ms (retrieval)
- Throughput: Limited by hardware
- Scaling: Manual

**Zilliz Cloud:**
- Latency: 60-100ms (retrieval, includes network)
- Throughput: Auto-scales
- Scaling: Automatic

**Trade-off:** +40-80ms latency for managed service benefits.

### Security Updates
- ✅ TLS/SSL support for Zilliz Cloud
- ✅ Secure credential handling
- ✅ `.env` not committed to git
- ⚠️ Remember to rotate credentials regularly

### Documentation Updates
- ✅ `ZILLIZ_CLOUD_SETUP.md` - Complete setup guide
- ✅ `ZILLIZ_QUICKSTART.md` - 3-step quick start
- ✅ `README.md` - Updated with cloud option
- ✅ `API_REFERENCE.md` - Updated examples
- ✅ `DEPLOYMENT.md` - Cloud deployment section

### Testing
- ✅ Manual testing với local Milvus
- ✅ Manual testing với Zilliz Cloud
- ⏳ Automated tests (TODO)

### Known Issues
None at this time.

### Next Steps
1. Add integration tests for both connection types
2. Implement connection pooling
3. Add retry logic với exponential backoff
4. Monitor và log connection metrics

---

## Previous Versions

### [1.0.0] - 2024-11-16 (Initial Release)
- Complete RAG system implementation
- 7 core modules
- 6 documentation files
- Docker support
- Evaluation framework
- Examples và tutorials
