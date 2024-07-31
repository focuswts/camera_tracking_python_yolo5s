import cv2
from discord_webhook import DiscordWebhook, DiscordEmbed

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send(self, image, timestamp):
        # Converta a imagem para o formato JPEG e depois para bytes
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        # Criar o webhook e adicionar o embed
        webhook = DiscordWebhook(url=self.webhook_url)
        embed = DiscordEmbed(title='Pessoa Detectada', description=f'Pessoa detectada em {timestamp}', color='03b2f8')
        embed.set_image(url='attachment://detection.jpg')
        webhook.add_embed(embed)
        webhook.add_file(file=img_bytes, filename='detection.jpg')

        # Enviar o webhook e verificar a resposta
        try:
            response = webhook.execute()
            print(f'Requisição HTTP enviada. Status: {response.status_code}')
            print(f'Resposta: {response.text}')
        except Exception as e:
            print(f'Erro ao enviar a requisição HTTP: {e}')
