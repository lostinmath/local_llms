"""Security and privacy example for Local LLMs.

To run this example:
1. Install the package: pip install -e ..
2. Run: python security_example.py
"""

import os
from local_llms.utils.security import DataEncryption
from local_llms.utils.data_processor import DataProcessor


def main():
    """Demonstrate security and privacy features."""
    
    print("="*50)
    print("SECURITY & PRIVACY EXAMPLE")
    print("="*50)
    
    # 1. Data Encryption
    print("\n1. DATA ENCRYPTION")
    print("-" * 50)
    
    # Initialize encryption
    encryption = DataEncryption()
    
    # Encrypt text
    sensitive_text = "This is my private data that should be encrypted."
    print(f"Original text: {sensitive_text}")
    
    encrypted = encryption.encrypt_text(sensitive_text)
    print(f"Encrypted: {encrypted[:50]}...")
    
    decrypted = encryption.decrypt_text(encrypted)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {decrypted == sensitive_text}")
    
    # 2. File Encryption
    print("\n2. FILE ENCRYPTION")
    print("-" * 50)
    
    # Create a test file
    test_file = "/tmp/sensitive_data.txt"
    with open(test_file, 'w') as f:
        f.write("This file contains sensitive information.\n")
        f.write("User: john.doe@example.com\n")
        f.write("API Key: sk-1234567890abcdef\n")
    
    print(f"Created test file: {test_file}")
    
    # Encrypt the file
    encrypted_file = encryption.encrypt_file(test_file)
    print(f"Encrypted file: {encrypted_file}")
    
    # Decrypt the file
    decrypted_file = encryption.decrypt_file(encrypted_file, "/tmp/decrypted_data.txt")
    print(f"Decrypted file: {decrypted_file}")
    
    # Verify contents
    with open(decrypted_file, 'r') as f:
        print(f"Decrypted contents:\n{f.read()}")
    
    # 3. Data Anonymization
    print("\n3. DATA ANONYMIZATION")
    print("-" * 50)
    
    data_processor = DataProcessor()
    
    text_with_pii = """
    My email is john.doe@example.com and my phone is 555-123-4567.
    My IP address is 192.168.1.1 and SSN is 123-45-6789.
    """
    
    print("Original text:")
    print(text_with_pii)
    
    anonymized = data_processor.anonymize_data(text_with_pii)
    print("\nAnonymized text:")
    print(anonymized)
    
    # 4. Data Hashing
    print("\n4. DATA HASHING")
    print("-" * 50)
    
    data_to_hash = "sensitive_user_data"
    hash_value = DataEncryption.hash_data(data_to_hash)
    print(f"Original: {data_to_hash}")
    print(f"SHA256 Hash: {hash_value}")
    
    # Verify hash is consistent
    hash_value2 = DataEncryption.hash_data(data_to_hash)
    print(f"Consistent: {hash_value == hash_value2}")
    
    # 5. Secure Deletion
    print("\n5. SECURE DELETION")
    print("-" * 50)
    
    # Create a temporary file
    temp_file = "/tmp/to_be_deleted.txt"
    with open(temp_file, 'w') as f:
        f.write("This file will be securely deleted.\n" * 100)
    
    print(f"Created temporary file: {temp_file}")
    print(f"File exists: {os.path.exists(temp_file)}")
    
    # Securely delete
    DataEncryption.secure_delete(temp_file)
    print(f"After secure deletion, file exists: {os.path.exists(temp_file)}")
    
    # Cleanup
    print("\n6. CLEANUP")
    print("-" * 50)
    for file in [test_file, encrypted_file, decrypted_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")
    
    print("\n" + "="*50)
    print("Security example completed!")
    print("="*50)
    print("\nKey takeaways:")
    print("- Always encrypt sensitive data before storage")
    print("- Use anonymization to remove PII from training data")
    print("- Hash data for integrity verification")
    print("- Use secure deletion for sensitive files")
    print("- Keep all data processing local for privacy")


if __name__ == "__main__":
    main()
