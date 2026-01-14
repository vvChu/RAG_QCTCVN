import os
import pytest
from unittest.mock import patch

class TestConfig:
    @pytest.fixture
    def mock_env(self):
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test_key',
            'MILVUS_HOST': 'localhost',
            'MILVUS_PORT': '19530'
        }, clear=True):
            yield

    def test_settings_load_success(self, mock_env):
        # We need to reload the module to pick up env vars because Settings is instantiated at module level
        import importlib
        import ccba_rag.core.settings as settings_module
        importlib.reload(settings_module)
        
        config = settings_module.settings
        assert config.gemini_api_key == 'test_key'
        assert config.milvus_host == 'localhost'
        assert config.milvus_port == '19530'
        assert config.milvus_hnsw_m == 8  # Default value

    def test_settings_missing_required(self):
        if os.path.exists('.env'):
            pytest.skip("Skipping test because .env file exists and interferes with Pydantic settings loading")
            
        with patch.dict(os.environ, {}, clear=True):
            # pydantic raises ValidationError
            from pydantic import ValidationError
            import importlib
            import ccba_rag.core.settings as settings_module
            
            with pytest.raises(ValidationError):
                importlib.reload(settings_module)

    def test_settings_type_conversion(self):
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test_key',
            'MILVUS_HNSW_M': '16',
            'BGE_USE_FP16': 'False'
        }):
            import importlib
            import ccba_rag.core.settings as settings_module
            importlib.reload(settings_module)
            
            config = settings_module.settings
            assert config.milvus_hnsw_m == 16
            assert config.bge_use_fp16 is False
