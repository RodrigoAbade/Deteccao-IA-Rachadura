# WallEye AI — Detecção e segmentação de rachaduras em concreto

Projeto de **visão computacional** voltado à identificação e segmentação de rachaduras em superfícies de concreto, utilizando **Python, OpenCV e YOLOv8 (Ultralytics)**. O trabalho reúne experimentos de treinamento de modelos, inferência em imagens e vídeo, integração com Raspberry Pi e mecanismos experimentais de alerta.

> **Status:** protótipo de pesquisa e desenvolvimento. As detecções não substituem inspeções técnicas ou avaliações de segurança estrutural.

## Objetivo

Investigar o uso de inteligência artificial para apoiar o monitoramento visual de estruturas de concreto, identificando regiões com possíveis rachaduras e permitindo o registro ou envio de alertas.

## Tecnologias

- **IA e visão computacional:** Python, YOLOv8, Ultralytics e OpenCV
- **Dispositivo embarcado:** Raspberry Pi, câmera CSI e Picamera2
- **Integração:** API externa, QR Code para pareamento e notificações por e-mail em um dos fluxos
- **Treinamento:** scripts para treinamento e retomada de experimentos

## Funcionalidades presentes no repositório

- **Treinamento e retomada:** `train.py` e `resume_train.py`
- **Inferência em imagem:** `predict.py`
- **Captura por câmera e alertas para API:** `predict2.py`
- **Captura CSI no Raspberry Pi e alertas por e-mail:** `testeLinux.py`
- **Pareamento do dispositivo por QR Code:** `main.py` e módulos em `src/api/`

Nos dois fluxos de vídeo, o código utiliza o limiar `conf >= 0.8` e intervalo de 300 segundos entre alertas. Em `predict2.py`, há uma função SMTP definida, mas ela não é chamada pelo laço atual.

## Arquitetura — visão simplificada

```text
Imagem / câmera
       |
       v
  Inferência YOLO
       |
       v
Detecção / segmentação
       |
       +--> Captura OpenCV --> Alerta com imagem para API externa
       |
       +--> Raspberry Pi + câmera CSI --> Alerta por e-mail

Pareamento por QR Code --> API externa --> Configuração local do dispositivo
```

Os fluxos acima representam alternativas de execução do protótipo, não um único pipeline obrigatório. O backend externo não está incluído neste repositório.

## Organização do código

| Caminho | Responsabilidade |
| --- | --- |
| `main.py` | Pareamento por QR Code |
| `predict.py` | Inferência em imagem |
| `predict2.py` | Inferência por câmera e alertas para API |
| `testeLinux.py` | Câmera CSI e alerta por e-mail |
| `train.py` / `resume_train.py` | Treinamento e retomada |
| `src/api/` | Leitura de QR Code, pareamento e configuração |
| `configs/device_config.yaml` | Identificador e endereço da API do dispositivo |
| `runs/` | Artefatos de experimentos anteriores |
| `data.yaml` | Configuração do dataset |
| `requirements.txt` | Dependências Python |
| `.env.example` | Exemplo de configuração sem credenciais reais |

## Instalação e preparação

1. Utilize uma versão do Python compatível com as dependências listadas em `requirements.txt`.
2. Crie e ative um ambiente virtual, se desejado.
3. Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

4. Disponibilize os pesos treinados no caminho esperado pelo script escolhido:
   - `testeLinux.py`: `best.pt` na raiz;
   - `predict2.py`: `runs/segment/train7/weights/best.pt`.

Os pesos não são instalados pelo `pip`. Ajuste os caminhos caso sua organização local seja diferente.

### Inferência em imagem

```bash
python predict.py
```

Verifique no script os caminhos de entrada e dos pesos antes da execução.

### Raspberry Pi: câmera CSI e e-mail

No Raspberry Pi OS, configure a câmera CSI e instale o Picamera2 separadamente; essa biblioteca não está em `requirements.txt`.

Copie `.env.example` para `.env`, configure remetente, senha SMTP e destinatário e, no Bash, carregue as variáveis:

```bash
cp .env.example .env
# Edite o arquivo .env antes de prosseguir.
set -a
source .env
set +a
python testeLinux.py
```

O arquivo `.env` não é carregado automaticamente. As variáveis exigidas incluem `WALLEYE_EMAIL_FROM`, `WALLEYE_SMTP_PASSWORD` e `WALLEYE_EMAIL_TO`; host e porta possuem valores padrão no exemplo.

### Pareamento e alertas via API

Com o serviço externo disponível:

1. Execute `python main.py`.
2. Apresente um QR Code contendo `id`, `codigo_pareador` e `api_url`.
3. Confira `configs/device_config.yaml`.
4. Execute `python predict2.py`.

## Pesquisa e divulgação

O WallEye AI integra um trabalho de pesquisa sobre detecção de fissuras e rachaduras com visão computacional, apresentado em formato de pôster no **XII SIINTEC 2026**, em Salvador, Bahia.

Os experimentos de treinamento podem ser consultados nos artefatos do projeto. Métricas de desempenho devem ser interpretadas conforme o conjunto de validação, a versão do modelo e a execução específica; não são equivalentes a uma garantia de desempenho em campo.

## Limitações e cuidados

- A confiança do modelo **não** é uma medida validada de gravidade ou risco estrutural.
- O comportamento com câmeras, iluminação, redes e superfícies diferentes requer validação.
- Os mecanismos de alerta são protótipos e não garantem monitoramento contínuo.
- Não publique senhas SMTP, tokens ou arquivos `.env`. Se credenciais já foram expostas, revogue-as no provedor: removê-las do arquivo atual não as elimina do histórico Git.

## Autor

**Rodrigo Abade** — desenvolvimento e experimentação com visão computacional.

[GitHub](https://github.com/RodrigoAbade) · [Portfólio](https://rodrigo-abade.vercel.app)
