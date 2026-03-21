# Data Dictionary for sales_data_sample.csv

This document provides an overview of the fields in the sales_data_sample.csv file, including descriptions and data types.

| Field Name         | Description                                        | Data Type   |
|-------------------|----------------------------------------------------|-------------|
| sale_id           | Unique identifier for each sale                    | Integer     |
| product_id        | Identifier for the product sold                    | Integer     |
| customer_id       | Identifier for the customer                        | Integer     |
| sale_date         | Date of the sale                                   | Date        |
| quantity          | Number of items sold                               | Integer     |
| price_per_item    | Price per item sold                                | Float       |
| total_sale_amount | Total amount for the sale (quantity * price)      | Float       |
| payment_method     | Method of payment (e.g., credit card, cash)      | String      |
| store_location    | Location of the store where the sale occurred      | String      |
| sales_rep_id      | Identifier for the sales representative involved    | Integer     |

## Notes
- Data types are specified as follows:
    - **Integer**: Whole numbers
    - **Float**: Decimal numbers
    - **String**: Text
    - **Date**: Date format (YYYY-MM-DD)
