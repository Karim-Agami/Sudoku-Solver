import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_application_1/components/custom_elecated_button.dart';
import 'package:image_picker/image_picker.dart';

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text("pick the image"),
          backgroundColor: Colors.amber,
        ),
        backgroundColor: Colors.black,
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
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
          ),
        ));
  }
}

Future<void> pickImage(ImageSource source) async {
  final picker = ImagePicker();
  final pickedFile = await picker.pickImage(source: source);

  if (pickedFile != null) {
    File imageFile = File(pickedFile.path);
  } else {
    // Handle the case when no image is selected
  }
}
