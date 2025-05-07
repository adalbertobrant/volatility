# 📈 Analisador de Volatilidade para Mercados BR e US 📊

[![Licença: The Unlicense](https://img.shields.io/badge/License-The_Unlicense-blue.svg)](https://unlicense.org/)

**Autor:** \[Adalberto Caldeira] 👨‍💻

## 🧐 Apresentação do Projeto

Este software é uma aplicação Streamlit projetada para analisar a volatilidade e fornecer informações relevantes sobre ações, com foco em mercados brasileiro (🇧🇷) e americano (🇺🇸). Ele busca auxiliar traders e investidores a obterem insights sobre a dinâmica de preços e opções de ativos.

**Justificativa:**

Este projeto foi desenvolvido com finalidade educacional e de aprendizado. Não tem como objetivo fornecer aconselhamento de investimento, e nenhuma recomendação de investimento é feita aqui. Os desenvolvedores não são consultores financeiros e não aceitam responsabilidade por quaisquer decisões financeiras ou perdas resultantes do uso deste software. Sempre consulte um consultor financeiro profissional antes de tomar qualquer decisão de investimento.

## ⚙️ Instalação

Siga estas etapas para configurar o ambiente:

1.  **Clonar o repositório:**

    ```bash
    git clone https://github.com/adalbertobrant/volatility
    cd volatility
    ```

2.  **Criar e ativar um ambiente virtual:**

    ```bash
    python3 -m venv trader  # Criar o ambiente
    source trader/bin/activate   # Ativar (Linux/macOS)
    # Ou
    trader\Scripts\activate  # Ativar (Windows)
    ```

3.  **Instalar as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Executar a aplicação Streamlit:**

    ```bash
    streamlit run volatilidade_analise.py
    ```

## 🚀 O que este software faz?

Este software é uma aplicação Streamlit projetada para analisar a volatilidade e fornecer informações relevantes sobre ações, com foco em mercados brasileiro (🇧🇷) e americano (🇺🇸). Ele busca auxiliar traders e investidores a obterem insights sobre a dinâmica de preços e opções de ativos.

**Funcionalidades Chave:**

* **Análise de Volatilidade:** Calcula a volatilidade histórica de ações utilizando o estimador de Yang-Zhang, que é mais preciso do que métodos tradicionais, especialmente em mercados com gaps de abertura.
* **Dados de Opções:** Obtém dados de opções de ações das APIs do yFinance e Finnhub, permitindo a análise de volatilidade implícita e outras métricas relacionadas a opções.
* **Estrutura a Termo da Volatilidade:** Constrói e interpola a estrutura a termo da volatilidade implícita, fornecendo uma visão de como a volatilidade varia com o tempo até o vencimento das opções.
* **Dados Fundamentalistas:** Apresenta dados fundamentalistas das empresas (obtidos do yFinance e Finnhub), como múltiplos, informações financeiras e eventos corporativos (datas de resultados, dividendos, etc.).
* **Interface Interativa:** Utiliza o Streamlit para fornecer uma interface web amigável, onde os usuários podem inserir tickers, visualizar gráficos e tabelas, e interagir com os resultados da análise.
* **Flexibilidade de Fontes de Dados:** Utiliza múltiplas fontes de dados (Alpha Vantage, yFinance, Finnhub) com lógica de fallback para aumentar a robustez e confiabilidade dos dados.
* **Screener Básico:** Inclui um screener simples para ações populares dos EUA, facilitando a análise rápida de ativos de interesse.
* **Logging Detalhado:** Registra todas as operações e potenciais erros em um arquivo de log, auxiliando na depuração e no acompanhamento do comportamento da aplicação.

**Em resumo:** Esta aplicação combina a análise de volatilidade histórica e implícita com dados fundamentalistas, tudo em uma interface interativa, para fornecer aos usuários uma visão abrangente dos ativos financeiros.

## 📄 Licença

Este projeto é licenciado sob a [The Unlicense](https://unlicense.org/).
Se quiser cite o autor ou não.
