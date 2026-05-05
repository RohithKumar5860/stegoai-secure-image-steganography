import urllib.request
import urllib.parse
import json
import io
import numpy as np
from PIL import Image
import base64

def post_json(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def post_multipart(url, fields, files):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = bytearray()
    for key, value in fields.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode('utf-8'))
    for key, (filename, content, content_type) in files.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode('utf-8'))
        body.extend(content)
        body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
    req = urllib.request.Request(url, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

# 1. Create a dummy image
img = np.random.randint(80, 220, (300, 300, 3), dtype=np.uint8)
buf = io.BytesIO()
Image.fromarray(img).save(buf, format='PNG')
img_bytes = buf.getvalue()

# 2. Test /generate-keys
print('Testing /generate-keys...')
keys = post_json('http://127.0.0.1:5000/generate-keys', {'label': 'test'})
if not keys.get('success'):
    print('Failed /generate-keys:', keys)
    exit(1)
pub_key = keys['public_key']
priv_key = keys['private_key']

# 3. Test /encode
print('Testing /encode...')
files = {'image': ('test.png', img_bytes, 'image/png')}
fields = {'message': 'Hello from API!', 'public_key': pub_key}
enc_result = post_multipart('http://127.0.0.1:5000/encode', fields, files)
if not enc_result.get('success'):
    print('Failed /encode:', enc_result)
    exit(1)

stego_b64 = enc_result['stego_image_b64']
stego_bytes = base64.b64decode(stego_b64)

# 4. Test /decode
print('Testing /decode...')
files = {'image': ('stego.png', stego_bytes, 'image/png')}
fields = {'private_key': priv_key}
dec_result = post_multipart('http://127.0.0.1:5000/decode', fields, files)
if not dec_result.get('success'):
    print('Failed /decode:', dec_result)
    exit(1)

print('Decode success!', dec_result)

# 5. Test /verify
print('Testing /verify...')
files = {'image': ('stego.png', stego_bytes, 'image/png')}
ver_result = post_multipart('http://127.0.0.1:5000/verify', {}, files)
print('Verify result:', ver_result)
print('ALL TESTS PASSED')
