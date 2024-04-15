<?php

$servername = "localhost";
$username = "root";
$password = "";
$database = "thesis";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$csvFile = 'twitter.csv';

if (($handle = fopen($csvFile, "r")) !== FALSE) {
    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) { 

        $dateString = $data[2]; 

        $postCreatedDateTime = DateTime::createFromFormat('D M d H:i:s O Y', $dateString);
        
        $postCreatedDateTime->modify('+6 years');
        
        $formattedDateTime = $postCreatedDateTime->format('Y-m-d H:i:s');
        
        $updateSql = "UPDATE depression SET created_at = ? WHERE post_id = ?";
        $stmt = $conn->prepare($updateSql);
        $stmt->bind_param("si", $formattedDateTime, $data[0]); 
        $stmt->execute();

        if ($stmt->error) {
            echo "Error: " . $stmt->error . "<br>";
        } else {
            echo "Data updated successfully<br>";
        }

        $stmt->close();
    }
    fclose($handle);
} else {
    echo "Error opening file<br>";
}

$conn->close();

?>