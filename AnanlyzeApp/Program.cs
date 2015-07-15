using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MongoDB;
using MongoDB.Driver;
using MongoDB.Bson;
using System.Diagnostics;

namespace AnanlyzeApp
{
    class Program
    {
        static void Main(string[] args)
        {   
            NumberOfPrimitivesInBlocks();
            /*
            Stopwatch timer = Stopwatch.StartNew();
            LineData();
            timer.Stop();
            Console.WriteLine("Obtain Linedata, ms: " + timer.ElapsedMilliseconds);
            Console.ReadLine();
 
            timer = Stopwatch.StartNew();
            TextData();
            timer.Stop();
            Console.WriteLine("Obtain Textdata, ms: " + timer.ElapsedMilliseconds);
            Console.ReadLine();
            */
        }

        static void NumberOfPrimitivesInBlocks()
        {
            MongoClient ClientMongo = new MongoClient();
            MongoDatabase DatabaseMongo = ClientMongo.GetServer().GetDatabase("geometry");
            var files = DatabaseMongo.GetCollection<BsonDocument>("files").Distinct("BlockId");

            FileStream fs = new FileStream(@"c:\data\blockCount.txt", FileMode.Truncate);
            using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
            {
                foreach (var file in files)
                {
                    string fileId = file.ToString();
                    QueryDocument filter = new QueryDocument("FileId", fileId);
                       long count = DatabaseMongo.GetCollection("objects").Find(filter).Count();
                    writer.WriteLine(fileId+";"+count);
                }
            }
        }

        static void TextData()
        {
            string className = "AcDbText";

            List<string> jsonObjs = GetObjectJsonByClassName(className);

            string fileName = @"c:\Data\TextData.csv";

            //https://msdn.microsoft.com/en-us/library/3aadshsx(v=vs.110).aspx
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileName, FileMode.CreateNew);
                using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
                {
                    writer.WriteLine("Data" + "; " + "Value");

                    for (int i = 0; i < jsonObjs.Count; i++)
                    {
                        string jsonObj = jsonObjs[i];
                        BsonDocument doc = BsonDocument.Parse(jsonObj);
                        string textString = doc["TextString"].ToString();

                        string position = string.Format(
                            "({0}, {1},{2})",
                            doc["Position"]["X"].ToString(),
                            doc["Position"]["Y"].ToString(),
                            doc["Position"]["Z"].ToString());

                        writer.WriteLine(position + "; " + textString);
                        Console.Clear();
                        Console.Write(i);
                    }
                }
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }

        }

        static void LineData()
        {
            List<string> jsonObjs = GetObjectJsonByClassName("AcDbLine");
            StreamWriter sw = new StreamWriter(@"C:\Data\LineData.csv");
            sw.WriteLine("Alignment" + "; " + "Length");

            for (int i = 0; i < jsonObjs.Count; i++)
            {
                string jsonObj = jsonObjs[i];
                BsonDocument doc = BsonDocument.Parse(jsonObj);

                double startX = doc["StartPoint"]["X"].ToDouble();
                double startY = doc["StartPoint"]["Y"].ToDouble();
                double endX = doc["EndPoint"]["X"].ToDouble();
                double endY = doc["EndPoint"]["Y"].ToDouble();
                double length = doc["Length"].ToDouble();

                string rotated = "Rotated";
                if (startX == endX)
                    rotated = "Vertical";
                if (startY == endY)
                    rotated = "Horizontal";

                sw.WriteLine(rotated + "; " + length);

                Console.Clear();
                Console.Write(i);
            }

            sw.Close();
        }

        static public List<string> GetObjectJsonByClassName(string className)
        {
            MongoClient ClientMongo = new MongoClient();
            MongoDatabase DatabaseMongo = ClientMongo.GetServer().GetDatabase("geometry");

            List<string> result = new List<string>();
            if (!string.IsNullOrEmpty(className))
            {
                QueryDocument filter = new QueryDocument("ClassName", className);
                var objJsons = DatabaseMongo.GetCollection<BsonDocument>("objects").Find(filter).SetLimit(1000000);
                foreach (var anObject in objJsons)
                    result.Add(anObject.ToString());
            }
            else
            {
                var objJsons = DatabaseMongo.GetCollection("objects").FindAll();
                foreach (var anObject in objJsons)
                    result.Add(anObject.ToString());
            }

            return result;
        }

    }
}
