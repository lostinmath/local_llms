"""Security and privacy utilities for data protection."""

import os
import hashlib
from pathlib import Path
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class DataEncryption:
    """Handles encryption and decryption of sensitive data."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize encryption handler.
        
        Args:
            key: Encryption key. If None, uses environment variable or generates new key.
        """
        if key is None:
            key = self._get_or_generate_key()
        
        self.cipher = Fernet(key)
        self._key = key
    
    @staticmethod
    def _get_or_generate_key() -> bytes:
        """Get encryption key from environment or generate a new one."""
        key_str = os.getenv("ENCRYPTION_KEY")
        
        if key_str:
            # Derive key from provided string
            return DataEncryption.derive_key_from_password(key_str)
        else:
            # Generate new key
            return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password string
            salt: Salt for key derivation. If None, uses fixed salt (not recommended for production)
            
        Returns:
            Derived encryption key
        """
        if salt is None:
            # WARNING: Fixed salt for consistent key generation from the same password.
            # This is ONLY secure for single-user systems where the same password
            # always generates the same key. For production multi-user systems,
            # generate a random salt and store it securely alongside encrypted data.
            # Example: salt = os.urandom(16)
            salt = b'local_llms_salt_v1'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        # Encode as base64 for Fernet
        import base64
        return base64.urlsafe_b64encode(key)
    
    def encrypt_text(self, text: str) -> bytes:
        """Encrypt text string.
        
        Args:
            text: Plain text to encrypt
            
        Returns:
            Encrypted bytes
        """
        return self.cipher.encrypt(text.encode('utf-8'))
    
    def decrypt_text(self, encrypted_data: bytes) -> str:
        """Decrypt encrypted data to text.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted text string
        """
        return self.cipher.decrypt(encrypted_data).decode('utf-8')
    
    def encrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Encrypt a file.
        
        Args:
            input_path: Path to file to encrypt
            output_path: Path for encrypted output. If None, appends .encrypted to input path
            
        Returns:
            Path to encrypted file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + '.encrypted')
        else:
            output_path = Path(output_path)
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Decrypt an encrypted file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted output. If None, removes .encrypted suffix
            
        Returns:
            Path to decrypted file
        """
        input_path = Path(input_path)
        if output_path is None:
            if input_path.suffix == '.encrypted':
                output_path = input_path.with_suffix('')
            else:
                output_path = input_path.with_suffix(input_path.suffix + '.decrypted')
        else:
            output_path = Path(output_path)
        
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path
    
    @staticmethod
    def hash_data(data: Union[str, bytes]) -> str:
        """Create SHA256 hash of data.
        
        Args:
            data: Data to hash (string or bytes)
            
        Returns:
            Hex string of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def secure_delete(file_path: Union[str, Path]) -> None:
        """Securely delete a file by overwriting before deletion.
        
        Args:
            file_path: Path to file to securely delete
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Overwrite with random data
        with open(file_path, 'wb') as f:
            f.write(os.urandom(file_size))
        
        # Delete the file
        file_path.unlink()
    
    def get_key(self) -> bytes:
        """Get the encryption key (use with caution).
        
        Returns:
            Encryption key bytes
        """
        return self._key
