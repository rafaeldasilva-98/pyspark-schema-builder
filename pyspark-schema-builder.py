from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    IntegerType,
    FloatType,
    DoubleType,
    BooleanType,
    LongType,
    ShortType,
    ByteType,
    TimestampType,
    DateType,
    BinaryType,
    ArrayType,
    MapType,
    DecimalType,
    DataType,
)

# Mapping of primitive types to PySpark classes
PRIMITIVE_TYPE_MAP: dict[str, type] = {
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
    """
    Represents the definition of a column with a name, type definition (as string),
    and whether the column is nullable.
    """
    name: str
    type_str: str
    nullable: bool = True


class ITypeParser(ABC):
    """
    Interface for type parsers. This abstract class defines the contract for parsing
    a type definition string into a PySpark DataType.
    """

    @abstractmethod
    def parse(self, type_str: str) -> DataType:
        """
        Parses a type definition string and returns a corresponding PySpark DataType.

        :param type_str: The string representation of the type.
        :return: A PySpark DataType object.
        """
        pass


class DefaultSparkTypeParser(ITypeParser):
    """
    Default implementation of ITypeParser using a recursive descent parser
    for converting a type definition string into a PySpark DataType.

    Supported types include primitives, array, map, struct (with nesting), and decimal.
    """

    def __init__(self, input_str: str) -> None:
        """
        Initializes the parser with the input string.

        :param input_str: The type definition string (e.g., "struct<id:int, info:struct<name:string>>").
        """
        self.input_str: str = input_str.strip()
        self.index: int = 0

    def _current_char(self) -> Union[str, None]:
        """
        Returns the current character or None if the end of the string is reached.
        """
        return self.input_str[self.index] if self.index < len(self.input_str) else None

    def _consume(self) -> str:
        """
        Consumes and returns the current character, advancing the index.
        """
        char: str = self.input_str[self.index]
        self.index += 1
        return char

    def _consume_whitespace(self) -> None:
        """
        Advances the index, skipping any whitespace characters.
        """
        while self._current_char() is not None and self._current_char().isspace():
            self._consume()

    def _expect(self, expected_char: str) -> None:
        """
        Ensures that the expected character is present at the current position.
        Raises a ValueError if the expected character is not found.

        :param expected_char: The character expected at the current position.
        """
        self._consume_whitespace()
        if self._current_char() != expected_char:
            raise ValueError(
                f"Expected '{expected_char}' at position {self.index} in: {self.input_str}"
            )
        self._consume()

    def _parse_number(self) -> int:
        """
        Parses an integer number from the current position.

        :return: The integer value parsed.
        """
        self._consume_whitespace()
        start_index: int = self.index
        while self._current_char() is not None and self._current_char().isdigit():
            self._consume()
        number_str: str = self.input_str[start_index:self.index]
        if not number_str:
            raise ValueError(
                f"Expected a number at position {start_index} in: {self.input_str}"
            )
        return int(number_str)

    def _parse_identifier(self) -> str:
        """
        Parses an identifier (a keyword or name) that can include letters, digits,
        underscores, or hyphens.

        :return: The parsed identifier in lower case.
        """
        self._consume_whitespace()
        start_index: int = self.index
        while (
            self._current_char() is not None
            and (self._current_char().isalnum() or self._current_char() in ['_', '-'])
        ):
            self._consume()
        if start_index == self.index:
            raise ValueError(
                f"Expected an identifier at position {self.index} in: {self.input_str}"
            )
        return self.input_str[start_index:self.index].lower()

    def _parse_fields(self) -> List[StructField]:
        """
        Parses the fields of a struct type, allowing for nested definitions.

        :return: A list of StructField objects.
        """
        fields: List[StructField] = []
        while True:
            self._consume_whitespace()
            if self._current_char() == ">":
                break
            field_name: str = self._parse_identifier()
            self._consume_whitespace()
            self._expect(":")
            field_data_type: DataType = self._parse_type()
            fields.append(StructField(field_name, field_data_type, True))
            self._consume_whitespace()
            if self._current_char() == ",":
                self._consume()  # consume the comma
            else:
                break
        return fields

    def _parse_type(self) -> DataType:
        """
        Recursively parses a type definition and returns the corresponding PySpark DataType.

        :return: The PySpark DataType corresponding to the definition.
        """
        self._consume_whitespace()
        type_identifier: str = self._parse_identifier()

        if type_identifier == "array":
            self._consume_whitespace()
            self._expect("<")
            element_type: DataType = self._parse_type()
            self._consume_whitespace()
            self._expect(">")
            return ArrayType(element_type)
        elif type_identifier == "map":
            self._consume_whitespace()
            self._expect("<")
            key_type: DataType = self._parse_type()
            self._consume_whitespace()
            self._expect(",")
            value_type: DataType = self._parse_type()
            self._consume_whitespace()
            self._expect(">")
            return MapType(key_type, value_type)
        elif type_identifier == "struct":
            self._consume_whitespace()
            self._expect("<")
            struct_fields: List[StructField] = self._parse_fields()
            self._consume_whitespace()
            self._expect(">")
            return StructType(struct_fields)
        elif type_identifier == "decimal":
            # Supports the format decimal(precision, scale)
            self._consume_whitespace()
            self._expect("(")
            precision: int = self._parse_number()
            self._consume_whitespace()
            self._expect(",")
            scale: int = self._parse_number()
            self._consume_whitespace()
            self._expect(")")
            return DecimalType(precision, scale)
        else:
            # If it's a primitive or an unmapped alias, return the corresponding DataType.
            if type_identifier in PRIMITIVE_TYPE_MAP:
                return PRIMITIVE_TYPE_MAP[type_identifier]()
            else:
                # Fallback to StringType for unmapped types.
                return StringType()

    def parse(self) -> DataType:
        """
        Performs a complete parse of the input string and returns the PySpark DataType.

        :return: The resulting PySpark DataType.
        :raises ValueError: If extra content remains after parsing.
        """
        data_type: DataType = self._parse_type()
        self._consume_whitespace()
        if self.index != len(self.input_str):
            raise ValueError(
                f"Extra content after parsing at position {self.index}: {self.input_str[self.index:]}"
            )
        return data_type

    # Implementation of the ITypeParser interface
    def parse(self, type_str: str = None) -> DataType:
        """
        Parses a type definition string and returns a corresponding PySpark DataType.
        If type_str is provided, it reinitializes the parser.

        :param type_str: Optional string representing the type definition.
        :return: A PySpark DataType.
        """
        if type_str is not None:
            self.input_str = type_str.strip()
            self.index = 0
        return self._parse_type()  # Use _parse_type() directly; no extra content check here


class SparkSchemaBuilder:
    """
    Builder class for constructing a PySpark schema from column definitions.
    
    Uses an injected ITypeParser to parse column type strings, making it flexible to adopt
    different parsing strategies if needed.
    """

    def __init__(self, columns: List[ColumnDefinition], type_parser: ITypeParser = None) -> None:
        """
        Initializes the schema builder with a list of column definitions and an optional type parser.

        :param columns: A list of ColumnDefinition objects.
        :param type_parser: An implementation of ITypeParser. Defaults to DefaultSparkTypeParser.
        """
        self.columns: List[ColumnDefinition] = columns
        # Use the default parser if none is provided; a new instance is created per parse
        self.type_parser: ITypeParser = type_parser or DefaultSparkTypeParser("")

    def _build_fields(self) -> List[StructField]:
        """
        Builds the list of StructField objects based on the column definitions.

        :return: A list of StructField objects.
        """
        fields: List[StructField] = []
        for col_def in self.columns:
            # Parse the data type using the injected type parser
            data_type: DataType = self.type_parser.parse(col_def.type_str)
            fields.append(StructField(col_def.name, data_type, col_def.nullable))
        return fields

    def execute(self) -> StructType:
        """
        Builds and returns the complete PySpark schema (StructType) from the column definitions.

        :return: A StructType representing the schema.
        """
        return StructType(self._build_fields())


# Example usage with complex nested types
if __name__ == "__main__":
    columns_definitions: List[ColumnDefinition] = [
        ColumnDefinition(name="id", type_str="int"),
        ColumnDefinition(name="name", type_str="string"),
        ColumnDefinition(name="tags", type_str="array<string>"),
        ColumnDefinition(name="metadata", type_str="map<string, int>"),
        ColumnDefinition(
            name="nested",
            type_str="struct<age:int, score:double, details:struct<active:boolean, last_login:timestamp, extra:array<struct<attr:string, value:decimal(10,2)>>> >"
        ),
    ]

    schema_builder: SparkSchemaBuilder = SparkSchemaBuilder(columns_definitions)
    schema: StructType = schema_builder.execute()
    print(schema)
