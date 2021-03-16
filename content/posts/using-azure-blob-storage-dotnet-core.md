---
title: "Using Azure blob storage in dotnet core"
date: 2020-08-20T18:48:46-04:00
showDate: true
draft: false
tags: ["blog","story"]
mermaid: false
---
There's a lot of mixed content on using Azure Blob storage but they tend to focus on using FileSteam to access and upload to a blob container.

#### Pre-requisites

-   Created a storage account in Azure - [https://docs.microsoft.com/en-gb/azure/storage/common/storage-account-create](https://docs.microsoft.com/en-gb/azure/storage/common/storage-account-create)

#### Add the package Azure.Storage.Blobs

```
dotnet add package Azure.Storage.Blobs
```

This is version 12+, older versions are called Microsoft.Azure.Storage.Blob

#### Get connection and setup a container

Get the connection string from the storage container you've created, it's in Access Keys. Copy Key 1 connection string. It will look something like:

_DefaultEndpointsProtocol=https;AccountName=xxxxx....._  
  
Whilst you're in the Azure, create a new container.

#### The code

FYI If your UploadAsync in BlobClient is hanging then it's because your memory stream is at the end, set its position =0.

```
var connString = "DefaultEndpointsProtocol=https;AccountName=xxxxx your key 1 connection";
var containerName ="Name of your container";
            
BlobContainerClient blobContainer = new BlobContainerClient(connString,containerName);
            
var someFileName ="hello.jpg";
//using a BlobClient
BlobClient client = blobContainer.GetBlobClient(someFileName);        

byte[] bytes; //from a file or other content
using (var writer = new MemoryStream())
{
  await writer.WriteAsync(bytes, 0, bytes.Length);
  writer.Position =0; //important when using memory stream else this will hang
  await client.UploadAsync(writer, true, CancellationToken.None);
}
            
using (var writer = new FileStream(Path.Combine("C:\\temp\\",someFileName),FileMode.Open))
{
  await client.UploadAsync(writer, true, CancellationToken.None);
}

//using BlobContainer            
await blobContainer.DeleteBlobIfExistsAsync(someFileName);
```

Enjoy