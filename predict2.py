import json
import cv2
from ultralytics import YOLO
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
import time
import os
import requests
import yaml 
with open('configs/device_config.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)


model_path = 'runs/segment/train7/weights/best.pt'
model = YOLO(model_path)

EMAIL_REMETENTE = os.environ.get("WALLEYE_EMAIL_FROM", "")
SENHA = os.environ.get("WALLEYE_SMTP_PASSWORD", "")
EMAIL_DESTINATARIO = os.environ.get("WALLEYE_EMAIL_TO", "")

SMTP_HOST = os.environ.get("WALLEYE_SMTP_HOST", "smtp.hostinger.com")
SMTP_PORT = int(os.environ.get("WALLEYE_SMTP_PORT", "587"))

DELAY_ALERTA = 300  
ultimo_alerta = 0

# ----------------------
# Função para enviar e-mail com anexo
# ----------------------
def enviar_email(imagem_path):
    try:
        if not all((EMAIL_REMETENTE, SENHA, EMAIL_DESTINATARIO)):
            raise ValueError(
                "Configure WALLEYE_EMAIL_FROM, WALLEYE_SMTP_PASSWORD e WALLEYE_EMAIL_TO."
            )
        msg = MIMEMultipart("alternative")
        msg['From'] = EMAIL_REMETENTE
        msg['To'] = EMAIL_DESTINATARIO
        msg['Subject'] = "⚠️ Alerta Automático — Rachadura Estrutural Detectada"

        corpo_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f4f6f8; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); padding: 20px;">
                <h2 style="color: #d9534f;">⚠️ Alerta de Segurança Estrutural</h2>
                <p>Prezados(as),</p>
                <p>O sistema de monitoramento <strong>Walleye AI</strong> detectou uma possível <strong>rachadura estrutural</strong> 
                com um índice de confiança superior a <strong>80%</strong>.</p>
                <p>O registro foi feito automaticamente e a imagem em anexo contém a evidência visual capturada no momento da detecção.</p>
                <p style="margin-top: 20px;">📅 <strong>Data e hora da detecção:</strong> {time.strftime("%d/%m/%Y %H:%M:%S")}</p>
                <hr style="border:none; border-top:1px solid #ddd; margin:20px 0;">
                <p style="color: #555;">Este é um e-mail automático enviado pelo sistema de vigilância inteligente da Walleye AI.</p>
                <p style="font-size: 13px; color: #777;">Por favor, não responda a esta mensagem.</p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(corpo_html, 'html'))

        with open(imagem_path, "rb") as f:
            mime = MIMEBase('image', 'jpeg')
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            mime.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(imagem_path)}"')
            msg.attach(mime)

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()

        print("Email enviado com sucesso!")

    except Exception as e:
        print("Erro ao enviar email:", e)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(" Não foi possível acessar a câmera.")
    exit()

FRAME_INTERVAL = 3 
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_INTERVAL != 0:
        continue  

    results = model(frame, task='segment', verbose=False, device="cpu")
    result = results[0]

    url = f"{data.get('api_url')}/alerts/create"
    url_email = f"{data.get('api_url')}/email/send"
    device_id = data.get('id')

    if result.masks and result.boxes:
        for i, m in enumerate(result.masks.data):
            conf = result.boxes.conf[i].item()

            if conf >= 0.8:
                tempo_atual = time.time()
                if tempo_atual - ultimo_alerta > DELAY_ALERTA:
                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    file_timestamp = time.strftime("%Y%m%d-%H%M%S")
                    img_name = f"alerta_{file_timestamp}.jpeg"
                    cv2.imwrite(img_name, frame)

                    ultimo_alerta = tempo_atual
                    dados = {"id_dispositivo": device_id, "date_detection": timestamp, "severity": 'alta', "messager": 'CASA DA MAE JOANA'}
                    files = {
                        "dto": (None, json.dumps(dados), "application/json"),
                        "file": (img_name, open(img_name, "rb"), "image/jpeg")
                    }
                    try:
                        print(f"📤 Enviando alerta com imagem {img_name}...")
                        resp = requests.post(url, files=files, timeout=10)
                        print("📡 Status:", resp.status_code)
                        print("📬 Resposta:", resp.text)
                    except Exception as e:
                        print("❌ Falha ao enviar alerta:", e)
                    break  

            mask_array = m.cpu().numpy()
            mask_color = np.zeros_like(frame, dtype=np.uint8)
            mask_color[:, :, 2] = (mask_array * 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cv2.putText(frame,
                        f"{conf*100:.1f}%",
                        (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

    cv2.imshow("Detecção de Rachaduras - Pressione 'q' para sair", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Sistema encerrado.")
