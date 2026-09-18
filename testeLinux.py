from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText

# ==============================
# CONFIGURAÇÕES GERAIS
# ==============================
MODEL_PATH = "best.pt"
DELAY_ALERTA = 300  # 5 minutos entre alertas
ultimo_alerta = 0

# ==============================
# CONFIGURAÇÕES DE E-MAIL (HOSTINGER)
# ==============================
EMAIL_REMETENTE = os.environ.get("WALLEYE_EMAIL_FROM", "")
SENHA = os.environ.get("WALLEYE_SMTP_PASSWORD", "")
EMAIL_DESTINATARIO = os.environ.get("WALLEYE_EMAIL_TO", "")

SMTP_HOST = os.environ.get("WALLEYE_SMTP_HOST", "smtp.hostinger.com")
SMTP_PORT = int(os.environ.get("WALLEYE_SMTP_PORT", "587"))

# ==============================
# FUNÇÃO DE ENVIO DE E-MAIL
# ==============================
def enviar_email(imagem_path):
    try:
        if not all((EMAIL_REMETENTE, SENHA, EMAIL_DESTINATARIO)):
            raise ValueError(
                "Configure WALLEYE_EMAIL_FROM, WALLEYE_SMTP_PASSWORD e WALLEYE_EMAIL_TO."
            )
        msg = MIMEMultipart()
        msg["From"] = EMAIL_REMETENTE
        msg["To"] = EMAIL_DESTINATARIO
        msg["Subject"] = "⚠️ Alerta Automático – Detecção de Rachadura Estrutural"

        corpo = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #222;">
            <h2 style="color: #B00020;">🚨 Alerta de Rachadura Detectada</h2>
            <p>Prezado responsável,</p>

            <p>O sistema automatizado de monitoramento estrutural <b>Walleye</b> detectou uma possível rachadura 
            em uma das superfícies monitoradas. A detecção foi realizada por meio de um modelo de Inteligência Artificial 
            com <b>confiança superior a 80%</b>.</p>

            <p>📅 <b>Data e hora da detecção:</b> {time.strftime("%d/%m/%Y %H:%M:%S")}<br>
               📸 <b>Imagem capturada:</b> em anexo</p>

            <p>Recomenda-se a verificação imediata do local indicado para avaliar a gravidade da anomalia e 
            realizar as devidas ações preventivas.</p>

            <p style="margin-top: 20px;">Atenciosamente,</p>
            <p><b>Equipe de Monitoramento Walleye</b><br>
            Sistema de Detecção de Rachaduras Estruturais<br>
            <a href="https://walleye.com.br" target="_blank">www.walleye.com.br</a></p>

            <hr style="margin-top: 30px;">
            <small style="color: gray;">Esta é uma mensagem automática gerada pelo sistema de monitoramento. 
            Não responda diretamente a este e-mail.</small>
        </body>
        </html>
        """

        msg.attach(MIMEText(corpo, "html"))

        # Anexo da imagem
        with open(imagem_path, "rb") as f:
            mime = MIMEBase("image", "jpeg")
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            mime.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(imagem_path)}"')
            msg.attach(mime)

        # Envio pelo servidor SMTP Hostinger
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()

        print("📩 Email enviado com sucesso!")

    except Exception as e:
        print("❌ Erro ao enviar email:", e)

# ==============================
# CARREGAR MODELO YOLO
# ==============================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo '{MODEL_PATH}' não encontrado!")
    exit(1)

model = YOLO(MODEL_PATH)

# ==============================
# INICIALIZAR CÂMERA
# ==============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("📸 Câmera iniciada com sucesso!")

# ==============================
# LOOP PRINCIPAL DE DETECÇÃO
# ==============================
cv2.namedWindow("Detecção de Rachaduras", cv2.WINDOW_NORMAL)

FRAME_INTERVAL = 3  # processa 1 a cada 3 frames (aumenta FPS)
frame_count = 0

while True:
    try:
        frame = picam2.capture_array()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue  # pula alguns frames para melhorar desempenho

        results = model(frame, task="segment", verbose=False)
        result = results[0]

        if result.masks and result.boxes:
            for i, m in enumerate(result.masks.data):
                conf = result.boxes.conf[i].item()

                # 🚨 Detecta rachadura com alta confiança
                if conf >= 0.8:
                    agora = time.time()
                    if agora - ultimo_alerta > DELAY_ALERTA:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        img_name = f"alerta_{timestamp}.jpg"
                        cv2.imwrite(img_name, frame)

                        enviar_email(img_name)
                        ultimo_alerta = agora
                        break  # evita múltiplos envios da mesma cena

                # Máscara vermelha para destacar rachaduras
                mask_array = m.cpu().numpy()
                mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[:, :, 2] = (mask_resized * 255).astype(np.uint8)  # canal vermelho
                frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

                # Caixa delimitadora + confiança
                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"{conf*100:.1f}%", (x1, max(y1-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Detecção de Rachaduras", frame)

        # Pressione "q" para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("⚠️ Erro no loop:", e)
        time.sleep(1)
        continue

# ==============================
# ENCERRAMENTO
# ==============================
picam2.stop()
cv2.destroyAllWindows()
print("✅ Encerrado com sucesso.")
