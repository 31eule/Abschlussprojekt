from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import json

KEY_FILE = "data/aes_key.bin"

def load_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read()
        if len(key) != 32:
            raise ValueError("Ungültige Schlüssellänge.")
        return key
    else:
        key = os.urandom(32)
        os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key

KEY = load_key()

def encrypt_data(data: bytes, key: bytes = KEY) -> bytes:
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return iv + encrypted

def decrypt_data(encrypted_data: bytes, key: bytes = KEY) -> bytes:
    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    padded_data = decryptor.update(encrypted) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data

def save_encrypted_json(filename: str, obj, key: bytes = KEY):
    json_data = json.dumps(obj).encode("utf-8")
    encrypted = encrypt_data(json_data, key)
    with open(filename, "wb") as f:
        f.write(encrypted)

def load_encrypted_json(filename: str, key: bytes = KEY):
    with open(filename, "rb") as f:
        encrypted = f.read()
    decrypted = decrypt_data(encrypted, key)
    return json.loads(decrypted.decode("utf-8"))