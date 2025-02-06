import unittest
from pyspark.sql.types import (
    StringType, IntegerType, ArrayType, MapType, StructType, StructField,
    BooleanType, TimestampType, DecimalType, DataType
)

class TestDefaultSparkTypeParser(unittest.TestCase):

    def setUp(self):
        # Configuração inicial para cada teste, se necessário
        pass

    def test_parse_primitive_types(self):
        parser = DefaultSparkTypeParser("int")
        result = parser.parse()
        self.assertIsInstance(result, IntegerType)

        parser = DefaultSparkTypeParser("string")
        result = parser.parse()
        self.assertIsInstance(result, StringType)

    def test_parse_array_type(self):
        parser = DefaultSparkTypeParser("array<int>")
        result = parser.parse()
        self.assertIsInstance(result, ArrayType)
        self.assertIsInstance(result.elementType, IntegerType)

    def test_parse_map_type(self):
        parser = DefaultSparkTypeParser("map<string, boolean>")
        result = parser.parse()
        self.assertIsInstance(result, MapType)
        self.assertIsInstance(result.keyType, StringType)
        self.assertIsInstance(result.valueType, BooleanType)

    def test_parse_struct_type(self):
        parser = DefaultSparkTypeParser("struct<name:string, age:int>")
        result = parser.parse()
        self.assertIsInstance(result, StructType)
        self.assertEqual(len(result.fields), 2)
        self.assertEqual(result.fields[0].name, "name")
        self.assertIsInstance(result.fields[0].dataType, StringType)
        self.assertEqual(result.fields[1].name, "age")
        self.assertIsInstance(result.fields[1].dataType, IntegerType)

    def test_parse_nested_struct_type(self):
        parser = DefaultSparkTypeParser(
            "struct<id:int, info:struct<name:string, active:boolean>>"
        )
        result = parser.parse()
        self.assertIsInstance(result, StructType)
        self.assertEqual(len(result.fields), 2)
        self.assertEqual(result.fields[1].name, "info")
        self.assertIsInstance(result.fields[1].dataType, StructType)
        nested_fields = result.fields[1].dataType.fields
        self.assertEqual(len(nested_fields), 2)
        self.assertEqual(nested_fields[0].name, "name")
        self.assertIsInstance(nested_fields[0].dataType, StringType)
        self.assertEqual(nested_fields[1].name, "active")
        self.assertIsInstance(nested_fields[1].dataType, BooleanType)

    def test_parse_decimal_type(self):
        parser = DefaultSparkTypeParser("decimal(10,2)")
        result = parser.parse()
        self.assertIsInstance(result, DecimalType)
        self.assertEqual(result.precision, 10)
        self.assertEqual(result.scale, 2)

    def test_parse_invalid_type_string(self):
        parser = DefaultSparkTypeParser("struct<name:string, age:int")
        with self.assertRaises(ValueError):
            parser.parse()

    def test_parse_unexpected_character(self):
        parser = DefaultSparkTypeParser("array<int>>")
        with self.assertRaises(ValueError):
            parser.parse()

    def test_parse_unknown_type(self):
        parser = DefaultSparkTypeParser("unknown<type>")
        result = parser.parse()
        self.assertIsInstance(result, StringType)  # Fallback para StringType

    def test_parse_identifier(self):
        parser = DefaultSparkTypeParser("valid_identifier")
        result = parser._parse_identifier()
        self.assertEqual(result, "valid_identifier")

        parser = DefaultSparkTypeParser("123invalid")
        with self.assertRaises(ValueError):
            parser._parse_identifier()

    def test_parse_number(self):
        parser = DefaultSparkTypeParser("123")
        result = parser._parse_number()
        self.assertEqual(result, 123)

        parser = DefaultSparkTypeParser("abc")
        with self.assertRaises(ValueError):
            parser._parse_number()

    def tearDown(self):
        # Limpeza após cada teste, se necessário
        pass

if __name__ == "__main__":
    unittest.main()
