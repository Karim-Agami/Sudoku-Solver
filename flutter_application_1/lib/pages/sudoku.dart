import 'package:flutter/material.dart';

class Sudoku extends StatelessWidget {
  final String _responseText;

  const Sudoku(this._responseText, {Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SudokuGrid(responseText: _responseText),
    );
  }
}

class SudokuGrid extends StatelessWidget {
  final String responseText;

  const SudokuGrid({Key? key, required this.responseText}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Convert the response text into a list of integers
    List<String> values = responseText.split(',');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Sudoku Grid'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: GridView.builder(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 9, // 9 columns for Sudoku
            crossAxisSpacing: 2.0,
            mainAxisSpacing: 2.0,
          ),
          itemCount: values.length,
          itemBuilder: (context, index) {
            return Container(
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: Colors.blueGrey,
                border: Border.all(color: Colors.white, width: 0.5),
              ),
              child: Text(
                values[index],
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
