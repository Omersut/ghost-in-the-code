import requests
import json

def test_mixtral():
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mixtral",
                "prompt": "Merhaba, çalışıyor musun? Lütfen kısa bir yanıt ver.",
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.7
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Yanıt:", result.get("response", "Yanıt alınamadı"))
            print("\nMixtral başarıyla çalışıyor!")
        else:
            print(f"Hata: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Bağlantı hatası: {str(e)}")
        print("\nOllama servisinin çalıştığından emin olun: ollama serve")

if __name__ == "__main__":
    test_mixtral() 