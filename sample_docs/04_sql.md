# SQL joins and aggregation

INNER JOIN returns matching rows from both tables. LEFT JOIN keeps all rows from the left table and attaches matching rows from the right table; unmatched right-side fields are NULL. Joining tables with multiple matching keys can multiply rows, so validate uniqueness and row counts before reporting totals.

GROUP BY groups rows for aggregate functions such as SUM, AVG and COUNT. WHERE filters rows before aggregation. HAVING filters groups after aggregation. Window functions such as ROW_NUMBER and LAG operate over a specified ordering and partition while preserving individual rows.
