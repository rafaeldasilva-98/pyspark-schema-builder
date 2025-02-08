from dataclasses import dataclass
from typing import List
from pyspark.sql.types import (
    StructType,
    StructField,
    DataType,
    StringType,
    IntegerType,
    FloatType,
    DoubleType,
    BooleanType,
    ArrayType,
    MapType,
    DecimalType,
    LongType,
    ShortType,
    ByteType,
    TimestampType,
    DateType,
    BinaryType,
)

PRIMITIVE_TYPES = {
    "string": StringType,
    "int": IntegerType,
    "integer": IntegerType,
    "float": FloatType,
    "double": DoubleType,
    "boolean": BooleanType,
    "long": LongType,
    "short": ShortType,
    "byte": ByteType,
    "timestamp": TimestampType,
    "date": DateType,
    "binary": BinaryType,
}

@dataclass
class ColumnDefinition:
    name: str
    type_str: str
    nullable: bool = True

class SchemaParser:
    def __init__(self, type_str: str):
        self.input = type_str.strip()
        self.pos = 0

    def parse(self) -> DataType:
        data_type = self._parse_data_type()
        self._skip_whitespace()
        if self.pos < len(self.input):
            raise ValueError(f"Unexpected characters at position {self.pos}: '{self.input[self.pos:]}'")
        return data_type

    def _current_char(self) -> str:
        return self.input[self.pos] if self.pos < len(self.input) else ''

    def _advance(self):
        self.pos += 1

    def _skip_whitespace(self):
        while self._current_char().isspace():
            self._advance()

    def _parse_identifier(self) -> str:
        self._skip_whitespace()
        start = self.pos
        while self._current_char().isalnum() or self._current_char() in ('_', '-'):
            self._advance()
        return self.input[start:self.pos].lower()

    def _parse_data_type(self) -> DataType:
        identifier = self._parse_identifier()
        
        if identifier == "array":
            return self._parse_array()
        elif identifier == "map":
            return self._parse_map()
        elif identifier == "struct":
            return self._parse_struct()
        elif identifier == "decimal":
            return self._parse_decimal()
        else:
            return PRIMITIVE_TYPES.get(identifier, StringType)()

    def _parse_array(self) -> DataType:
        self._expect("<")
        element_type = self._parse_data_type()
        self._expect(">")
        return ArrayType(element_type)

    def _parse_map(self) -> DataType:
        self._expect("<")
        key_type = self._parse_data_type()
        self._expect(",")
        value_type = self._parse_data_type()
        self._expect(">")
        return MapType(key_type, value_type)

    def _parse_struct(self) -> DataType:
        self._expect("<")
        fields = []
        while self._current_char() != ">":
            field_name = self._parse_identifier()
            self._expect(":")
            field_type = self._parse_data_type()
            fields.append(StructField(field_name, field_type, True))
            if self._current_char() == ",":
                self._advance()
        self._expect(">")
        return StructType(fields)

    def _parse_decimal(self) -> DataType:
        self._expect("(")
        precision = self._parse_number()
        self._expect(",")
        scale = self._parse_number()
        self._expect(")")
        return DecimalType(precision, scale)

    def _parse_number(self) -> int:
        self._skip_whitespace()
        start = self.pos
        while self._current_char().isdigit():
            self._advance()
        return int(self.input[start:self.pos])

    def _expect(self, char: str):
        self._skip_whitespace()
        if self._current_char() != char:
            raise ValueError(f"Expected '{char}' at position {self.pos}")
        self._advance()

def create_schema(columns: List[ColumnDefinition]) -> StructType:
    fields = []
    for col in columns:
        parser = SchemaParser(col.type_str)
        data_type = parser.parse()
        fields.append(StructField(col.name, data_type, col.nullable))
    return StructType(fields)

columns = [
    ColumnDefinition("id", "int", False),
    ColumnDefinition("name", "string"),
    ColumnDefinition("scores", "array<double>"),
    ColumnDefinition("metadata", "struct<timestamp:timestamp,info:map<string,string>>")
]

schema = create_schema(columns)
schema.printTree()