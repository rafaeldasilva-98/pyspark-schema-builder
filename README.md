# PySpark Schema Builder

A robust and extensible solution for dynamically generating PySpark schemas (StructType) from type definition strings. Supports complex nested structures, arrays, maps, decimals, and all primitive Spark SQL types.

## Features

- **Primitive Type Support**: All Spark SQL primitives (`int`, `string`, `timestamp`, etc.)
- **Complex Types**: 
  - Nested `struct` (with unlimited nesting)
  - `array<element_type>`
  - `map<key_type,value_type>`
  - `decimal(precision,scale)`
- **Extensible Architecture**: Custom type parsers via `ITypeParser` interface
- **Nullability Control**: Explicit column nullability specification
- **Validation**: Syntax validation with detailed error messages

## Installation

Requires PySpark 3.x+. Add to dependencies:
```bash
pip install pyspark
