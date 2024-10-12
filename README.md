# Dashboard Interativo de Previsão de Vendas - Supermercado

Este projeto utiliza **Streamlit** para criar um dashboard interativo que realiza previsões de faturamento (vendas) de um supermercado utilizando um modelo de **Multi Layer Perceptron (MLP)**. O dashboard permite a visualização de dados históricos de vendas e faz previsões para os próximos três meses.

## Funcionalidades

- Exibe os dados históricos de vendas de um supermercado.
- Visualiza as vendas agregadas por mês em um gráfico.
- Realiza a previsão de vendas dos próximos três meses usando um modelo de Machine Learning (MLP).
- Exibe o **Erro Absoluto Médio (MAE)** da previsão.

## Tecnologias Utilizadas

- **Python 3.x**
- **Streamlit**: Framework para a criação de dashboards interativos.
- **Pandas**: Para manipulação e agregação dos dados.
- **Matplotlib**: Para plotar gráficos das vendas mensais.
- **Scikit-learn**: Para treinar o modelo MLP e realizar as previsões.

## Instalação

### Pré-requisitos

Certifique-se de ter o **Python** instalado em seu ambiente. Além disso, você precisará instalar as dependências listadas no arquivo `requirements.txt`.

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```

2. Navegue até o diretório do projeto:

   ```bash
   cd seu-repositorio
   ```

3. Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows, use venv\Scripts\activate
   ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Dependências

Se o arquivo `requirements.txt` não estiver disponível, instale as dependências manualmente:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

## Execução

1. Certifique-se de que o arquivo CSV com os dados de vendas do supermercado esteja no mesmo diretório do projeto ou ajuste o caminho no código conforme necessário.
2. Execute o Streamlit:

   ```bash
   streamlit run streamlit_dashboard.py
   ```

3. O Streamlit abrirá automaticamente o navegador padrão com o dashboard.

## Estrutura do Projeto

```bash
├── regression_mlp.py           # Arquivo principal contendo o código do MLP
├── streamlit_dashboard.py      # Arquivo principal contendo o código do dashboard
├── supermarket_sales.csv       # Arquivo CSV com os dados históricos de vendas
├── README.md                   # Este arquivo README
└── requirements.txt            # Lista de dependências do projeto
```

## Explicação do Código

O código principal está no arquivo `dashboard.py` e contém as seguintes seções:

- **Carregamento de dados**: Carrega os dados do arquivo CSV e os formata para o uso no dashboard.
- **Visualização de vendas**: Gera um gráfico de vendas mensais, agregando as vendas por mês.
- **Treinamento do Modelo**: Treina um modelo de MLP (Multi Layer Perceptron) utilizando as vendas mensais para prever os próximos três meses.
- **Previsão de Vendas**: Realiza a previsão do faturamento para os próximos três meses e exibe os resultados no dashboard.

## Contribuindo

Sinta-se à vontade para contribuir com melhorias ou novas funcionalidades! Para contribuir:

1. Faça um fork do repositório.
2. Crie uma nova branch para sua feature:
   ```bash
   git checkout -b nova-feature
   ```
3. Faça commit de suas alterações:
   ```bash
   git commit -m 'Adicionar nova feature'
   ```
4. Envie as alterações para o repositório remoto:
   ```bash
   git push origin nova-feature
   ```
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato

- [@renanalencar](htt)
