<?php
 
$servername = "localhost";
$username = "root";
$password = "";
$database = "thesis";
 
$conn = new mysqli($servername, $username, $password, $database);
 
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
 
$csvFile = 'Mental-Health-Twitter/Mental-Health-Twitter.csv';
 
if (($handle = fopen($csvFile, "r")) !== FALSE) {
     
    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) { 
        $sql = "INSERT INTO depression (post_id, post_created, created_at, post_text, user_id, followers, friends, favourites, statuses, retweets, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
         
        $stmt = $conn->prepare($sql);
         
        $stmt->bind_param("sssssssssss", $data[0], $data[1], $data[2], $data[3], $data[4], $data[5], $data[6], $data[7], $data[8], $data[9], $data[10]);
         
        $stmt->execute();
         
        if ($stmt->error) {
            echo "Error: " . $stmt->error . "<br>";
        } else {
            echo "Data inserted successfully<br>";
        }
         
        $stmt->close();
    } 
    fclose($handle);
} else {
    echo "Error opening file<br>";
}
 
$conn->close();
?>