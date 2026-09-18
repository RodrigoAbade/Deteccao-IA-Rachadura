# Walleye — Detecção e segmentação de rachaduras

Projeto de visão computacional com Python, OpenCV e YOLO (Ultralytics). O repositório reúne treinamento, inferência, pareamento de dispositivo e duas alternativas de captura de vídeo.

## O que está implementado

- `main.py`: lê um QR Code, solicita pareamento à API e salva a configuração do dispositivo.
- `predict2.py`: usa câmera via OpenCV e envia alertas com imagem para uma API externa. Há uma função SMTP no arquivo, mas o laço atual não a chama.
- `testeLinux.py`: usa câmera CSI com Picamera2 no Raspberry Pi e envia alertas por e-mail.
- Nos dois scripts de vídeo, o limiar está definido no código como `conf >= 0.8` e o intervalo entre alertas é de 300 segundos.
- `train.py` e `resume_train.py`: treinamento e retomada do modelo.
- `predict.py`: inferência em imagem.

A confiança do modelo não representa uma classificação validada de gravidade estrutural. O repositório contém protótipos; disponibilidade contínua e desempenho precisam ser avaliados no equipamento de destino.

## Estrutura real

| Caminho | Finalidade |
| --- | --- |
| `main.py` | Pareamento por QR Code |
| `predict.py` / `predict2.py` | Inferência em imagem / câmera com API |
| `testeLinux.py` | Câmera CSI e alerta por e-mail |
| `train.py` / `resume_train.py` | Treinamento |
| `src/api/` | Leitor QR, cliente de pareamento e configuração |
| `configs/device_config.yaml` | ID e URL da API do dispositivo |
| `runs/` | Artefatos de treinamentos anteriores |
| `data.yaml` | Configuração do dataset |
| `requirements.txt` | Dependências Python |
| `.env.example` | Exemplo sem credenciais reais |

## Preparação

1. Crie um ambiente Python compatível com as versões de `requirements.txt` e instale as dependências: `python -m pip install -r requirements.txt`.
2. Para `testeLinux.py`, prepare também Picamera2 e a câmera CSI no Raspberry Pi OS. Picamera2 não está incluído em `requirements.txt`.
3. Disponibilize os pesos treinados: `testeLinux.py` espera `best.pt` na raiz; `predict2.py` usa `runs/segment/train7/weights/best.pt`. Ajuste o caminho no script se necessário. Os pesos não são instalados pelo pip.
4. Execute os comandos a partir da raiz do repositório.

### E-mail no Raspberry Pi

Copie `.env.example` para `.env` e preencha remetente, senha SMTP e destinatário. No Bash, carregue as variáveis antes de iniciar:

```bash
cp .env.example .env
# Edite .env localmente antes de continuar.
set -a
source .env
set +a
python testeLinux.py
```

O arquivo `.env` não é carregado automaticamente. Use aspas simples nos valores Bash, especialmente em senhas com caracteres especiais. O envio exige `WALLEYE_EMAIL_FROM`, `WALLEYE_SMTP_PASSWORD` e `WALLEYE_EMAIL_TO`; host e porta possuem os padrões do exemplo.

### Pareamento e alertas via API

Com a API externa disponível, execute `python main.py` e apresente um QR Code contendo `id`, `codigo_pareador` e `api_url`. Após o pareamento, confira `configs/device_config.yaml` e execute `python predict2.py`. O backend externo não faz parte deste repositório.

## Credenciais

Não versione senhas ou arquivos `.env`. Se uma senha SMTP já foi publicada, substitua-a no provedor antes de reutilizar o sistema. Retirar a senha da versão atual não a remove do histórico Git.
