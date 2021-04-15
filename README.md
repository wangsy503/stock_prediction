## 数据获取
*使用Tushare大数据开放社区的接口。文档请参照 https://waditu.com/document/2*


`generate_fund_list.ipynb`
用于获取在 20180101-20201231 期间存活的所有基金列表，其 output 为 `fund_list.csv`.


`generate_fund_manager.ipynb`
用于获取 `fund_list` 中所有基金 2018-2020 在职的基金经理人列表，其 output 为 `manager_list_whole.csv`.


`generate_portfolio.ipynb`
获取 `fund_list` 中所有基金在 2018-2020 公布的持仓比例，其 output 为 `raw_portfolio_list.csv` （前500个）和 `raw_portfolio_list_whole.csv` （所有）.


`portfolio_average.ipynb`
将 `raw_portfolio_list_whole.csv` 中基金所持股票按季度报告求平均， 其 output 为 `sum_fund_stock.csv`.


`generate_stock_table.ipynb`
临时文件，做一个效果看看.



## TODO
- 使用 AKShare 可以获取股票占基金的净值比例，please refer to `test.ipynb` for more information.