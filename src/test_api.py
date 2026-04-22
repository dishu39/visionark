import requests

url = "http://localhost:10000/caption"

# Use COCO image
image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
response = requests.get(image_url)

files = {'file': ('image.jpg', response.content, 'image/jpeg')}
data = {'auto_denoise': 'true'}

print("Sending request to API...")
res = requests.post(url, files=files, data=data)
print("Status Code:", res.status_code)
print("Response:", res.text)
