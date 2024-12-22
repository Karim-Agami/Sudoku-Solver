import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_application_1/components/custom_elecated_button.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  String _responseText = "";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Pick and Upload Image"),
        backgroundColor: Colors.amber,
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: () {
              setState(() {
                _image = null;
                _responseText = "";
              });
            },
          ),
        ],
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Display image preview if selected
              if (_image != null)
                Image.file(
                  _image!,
                  height: 200,
                  width: 200,
                  fit: BoxFit.cover,
                ),
              SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  CustomElevatedButton(
                    text: "Camera",
                    onPressed: () {
                      pickImage(ImageSource.camera);
                    },
                  ),
                  CustomElevatedButton(
                    text: "Gallery",
                    onPressed: () {
                      pickImage(ImageSource.gallery);
                    },
                  ),
                ],
              ),
              SizedBox(height: 20),
              if (_image != null)
                CustomElevatedButton(
                  text: "Upload",
                  onPressed: _uploadImage,
                ),
              SizedBox(height: 20),
              if (_responseText.isNotEmpty)
                Text(
                  _responseText,
                  style: TextStyle(color: Colors.white),
                  textAlign: TextAlign.center,
                ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _responseText = ""; // Reset response text on new image selection
      });
    } else {
      print("No image selected.");
    }
  }

  Future<void> _uploadImage() async {
    if (_image == null) return;

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://192.168.17.28:8000/upload'),
    );
    request.files.add(await http.MultipartFile.fromPath('file', _image!.path));

    final response = await request.send();
    if (response.statusCode == 200) {
      print("Image uploaded successfully.");
      final responseData = await response.stream.bytesToString();
      setState(() {
        _responseText = responseData; // Set the response text to display
        List<List<int>> matrix = stringToMatrix(responseData);
        print(matrix);
      });
    } else {
      setState(() {
        _responseText = 'Upload failed with status: ${response.statusCode}';
      });
    }
  }

  List<List<int>> stringToMatrix(String predictionStr) {
    // Step 1: Remove non-digit characters (like spaces, commas, or newlines)
    String cleanedStr = predictionStr.replaceAll(RegExp(r'\D'), '');

    // Step 2: Ensure the cleaned string has the right length (81 characters for a 9x9 matrix)
    if (cleanedStr.length != 81) {
      throw FormatException(
          'Invalid matrix size: expected 81 characters, got ${cleanedStr.length}.');
    }

    // Step 3: Convert the cleaned string into a list of integers
    List<int> flatMatrix =
        cleanedStr.split('').map((e) => int.parse(e)).toList();

    // Step 4: Reshape the flat list into a 9x9 matrix
    List<List<int>> matrix = [];
    for (int i = 0; i < 9; i++) {
      matrix.add(flatMatrix.sublist(i * 9, (i + 1) * 9));
    }

    return matrix;
  }
}
