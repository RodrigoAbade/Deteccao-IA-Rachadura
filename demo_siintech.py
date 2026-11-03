"""Demonstração local: câmera, segmentação e envio opcional por e-mail."""
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODEL = BASE / "models" / "train7" / "weights" / "best.pt"
OUTPUT = BASE / "resultados_demo"
OUTPUT.mkdir(exist_ok=True)


def send_email(image_path: Path, confidence: float) -> None:
    sender = os.getenv("WALLEYE_EMAIL_FROM", "").strip()
    recipient = os.getenv("WALLEYE_EMAIL_TO", "").strip()
    password = os.getenv("WALLEYE_SMTP_PASSWORD", "")
    host = os.getenv("WALLEYE_SMTP_HOST", "smtp.hostinger.com")
    port = int(os.getenv("WALLEYE_SMTP_PORT", "587"))
    if not all((sender, recipient, password)):
        raise RuntimeError("Configure WALLEYE_EMAIL_FROM, WALLEYE_EMAIL_TO e WALLEYE_SMTP_PASSWORD.")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = "Walleye AI - Resultado da análise de rachaduras"
    msg.set_content(
        f"Imagem analisada em {datetime.now():%d/%m/%Y %H:%M:%S}.\n"
        f"Maior confiança de detecção: {confidence:.1%}.\n"
        "O resultado é experimental e não substitui inspeção estrutural."
    )
    msg.add_attachment(image_path.read_bytes(), maintype="image", subtype="jpeg", filename=image_path.name)
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(sender, password)
        smtp.send_message(msg)


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL}")
    model = YOLO(str(MODEL))
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam. Verifique permissões e aplicativos usando a câmera.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    preview = "Walleye AI | ESPACO: analisar | Q: sair"
    print("Pressione ESPACO para capturar e analisar. Pressione Q para sair.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Falha ao ler a câmera.")
                break
            cv2.putText(frame, "ESPACO: analisar | Q: sair", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(preview, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 32:
                continue
            print("Analisando imagem na CPU...")
            result = model.predict(source=frame, device="cpu", imgsz=640, conf=0.25, verbose=False)[0]
            annotated = result.plot()
            confidence = float(result.boxes.conf.max().item()) if result.boxes is not None and len(result.boxes) else 0.0
            path = OUTPUT / f"analise_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if not cv2.imwrite(str(path), annotated):
                print("Falha ao salvar resultado.")
                continue
            cv2.imshow("Walleye AI | Resultado", annotated)
            cv2.waitKey(1)
            print(f"Imagem salva em: {path}")
            print(f"Detecções: {len(result.boxes) if result.boxes is not None else 0} | confiança máxima: {confidence:.1%}")
            if not os.getenv("WALLEYE_EMAIL_FROM"):
                print("E-mail não configurado. A imagem foi salva localmente.")
                continue
            try:
                send_email(path, confidence)
                print("E-mail enviado com sucesso!")
            except Exception as exc:
                print(f"Não foi possível enviar e-mail: {exc}. A imagem continua salva localmente.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
