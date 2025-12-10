"""Tests for security utilities."""

import os
import tempfile
import pytest
from pathlib import Path
from local_llms.utils.security import DataEncryption


def test_text_encryption_decryption():
    """Test text encryption and decryption."""
    encryption = DataEncryption()
    
    original_text = "This is a secret message!"
    encrypted = encryption.encrypt_text(original_text)
    decrypted = encryption.decrypt_text(encrypted)
    
    assert isinstance(encrypted, bytes)
    assert decrypted == original_text
    assert encrypted != original_text.encode()


def test_file_encryption_decryption():
    """Test file encryption and decryption."""
    encryption = DataEncryption()
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        test_file = f.name
        f.write("Secret file content\n")
        f.write("Line 2\n")
    
    try:
        # Encrypt file
        encrypted_file = encryption.encrypt_file(test_file)
        assert encrypted_file.exists()
        
        # Decrypt file
        decrypted_file = encryption.decrypt_file(encrypted_file)
        assert decrypted_file.exists()
        
        # Verify content
        with open(decrypted_file, 'r') as f:
            content = f.read()
        
        assert "Secret file content" in content
        assert "Line 2" in content
        
        # Cleanup
        os.remove(encrypted_file)
        os.remove(decrypted_file)
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_data_hashing():
    """Test data hashing."""
    data = "test data"
    hash1 = DataEncryption.hash_data(data)
    hash2 = DataEncryption.hash_data(data)
    
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 produces 64 hex characters
    assert hash1 == hash2  # Same input produces same hash
    
    # Different data produces different hash
    hash3 = DataEncryption.hash_data("different data")
    assert hash1 != hash3


def test_derive_key_from_password():
    """Test key derivation from password."""
    password = "my_secure_password"
    key1 = DataEncryption.derive_key_from_password(password)
    key2 = DataEncryption.derive_key_from_password(password)
    
    assert isinstance(key1, bytes)
    assert key1 == key2  # Same password produces same key
    
    # Different password produces different key
    key3 = DataEncryption.derive_key_from_password("different_password")
    assert key1 != key3


def test_secure_delete():
    """Test secure file deletion."""
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        test_file = f.name
        f.write("This file will be securely deleted\n" * 100)
    
    assert os.path.exists(test_file)
    
    # Securely delete
    DataEncryption.secure_delete(test_file)
    
    assert not os.path.exists(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
