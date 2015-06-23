using Crawl;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AnanlyzeApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //LineData();
            LineIsArectangle();
        }

        class DicList : Dictionary<object, List<object>>
        {
            public void AddToList(object key, object value)
            {
                List<object> currList;

                if (this.ContainsKey(key))
                {
                    currList = this[key];
                }
                else
                {
                    currList = new List<object>();
                }
                currList.Add(value);
            }
        }

        class Rectangle
        {
            public crawlAcDbLine line1;
            public crawlAcDbLine line2;
            public crawlAcDbLine line3;
            public crawlAcDbLine line4;
            private Angle angleA;
            private Angle angleC;

            public Rectangle(Angle angleA, Angle angleC)
            {
                this.angleA = angleA;
                this.angleC = angleC;

                this.line1 = angleA.line1;
                this.line4 = angleA.line2;

                this.line2 = angleC.line1;
                this.line3 = angleC.line2;
            }
        }

        class Angle
        {
            public crawlAcDbLine line1;
            public crawlAcDbLine line2;
            public crawlPoint3d point1;
            public crawlPoint3d point2;
            public crawlPoint3d pntCenter;

            public Angle(crawlAcDbLine leftLine, crawlAcDbLine rightLine)
            {
                this.line1 = leftLine;
                this.line2 = rightLine;

                if (leftLine.EndPoint.Equals(rightLine.EndPoint))
                {
                    pntCenter = leftLine.EndPoint;
                    point1 = leftLine.StartPoint;
                    point2 = rightLine.StartPoint;
                    return;
                }

                if (leftLine.EndPoint.Equals(rightLine.StartPoint))
                {
                    pntCenter = leftLine.EndPoint;
                    point1 = leftLine.StartPoint;
                    point2 = rightLine.EndPoint;
                    return;
                }

                if (leftLine.StartPoint.Equals(rightLine.EndPoint))
                {
                    pntCenter = leftLine.StartPoint;
                    point1 = leftLine.EndPoint;
                    point2 = rightLine.StartPoint;
                    return;
                }

                if (leftLine.StartPoint.Equals(rightLine.StartPoint))
                {
                    pntCenter = leftLine.StartPoint;
                    point1 = leftLine.EndPoint;
                    point2 = rightLine.EndPoint;
                    return;
                }

                throw new System.Exception();
            }
        }

        private static void AddLineToDict(Dictionary<string, List<crawlAcDbLine>> lines, crawlAcDbLine line)
        {
            string point = line.StartPoint.ToString();
            List<crawlAcDbLine> listOfLines;

            if (lines.ContainsKey(point))
            {
                listOfLines = lines[point];
            }
            else
            {
                listOfLines = new List<crawlAcDbLine>();
                lines.Add(point, listOfLines);
            }

            listOfLines.Add(line);

            point = line.EndPoint.ToString();
            if (lines.ContainsKey(point))
            {
                listOfLines = lines[point];
            }
            else
            {
                listOfLines = new List<crawlAcDbLine>();
                lines.Add(point, listOfLines);
            }

            listOfLines.Add(line);
        }

        static void LineIsArectangle()
        {
            SqlDb sqlDB = new SqlDb();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            StreamWriter sw = new StreamWriter(@"c:\Data\LineData.csv");
            sw.WriteLine("Alignment" + "; " + "Length; " + "FormsRectangle");

            List<crawlAcDbLine> lines = new List<crawlAcDbLine>();
            Dictionary<string, List<crawlAcDbLine>> pointDict = new Dictionary<string, List<crawlAcDbLine>>();
            Dictionary<string, List<Angle>> anglesDict = new Dictionary<string, List<Angle>>();

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                lines.Add(cLine);
                AddLineToDict(pointDict, cLine);
            }

            foreach (string point1 in pointDict.Keys)
            {
                if (pointDict[point1].Count > 1)
                {
                    crawlAcDbLine lineA = pointDict[point1][0];
                    crawlAcDbLine lineB = pointDict[point1][1];

                    Angle angle = new Angle(lineA, lineB);

                    List<Angle> listA = new List<Angle>();
                    if (anglesDict.ContainsKey(angle.pntCenter.ToString()))
                        listA = anglesDict[angle.pntCenter.ToString()];
                    else
                        anglesDict.Add(angle.pntCenter.ToString(), listA);

                    listA.Add(angle);
                }
            }
            List<Rectangle> rectangles = new List<Rectangle>();

            foreach (string pnt in anglesDict.Keys)
                foreach (Angle angleB in anglesDict[pnt])

                    if (anglesDict.ContainsKey(angleB.point1.ToString()))
                        if (anglesDict.ContainsKey(angleB.point2.ToString()))
                            foreach (Angle angleA in anglesDict[angleB.point1.ToString()])
                                foreach (Angle angleC in anglesDict[angleB.point2.ToString()])
                                {
                                    if (angleA.point2.Equals(angleB.pntCenter))
                                        if (angleA.point1.Equals(angleC.point1) || angleA.point1.Equals(angleC.point2))
                                        {
                                            Rectangle rectangle = new Rectangle(angleA, angleC);
                                            rectangles.Add(rectangle);
                                        }

                                    if (angleA.point1.Equals(angleB.pntCenter))
                                        if (angleA.point2.Equals(angleC.point1) || angleA.point2.Equals(angleC.point2))
                                        {
                                            Rectangle rectangle = new Rectangle(angleA, angleC);
                                            rectangles.Add(rectangle);
                                        }
                                }

            HashSet<crawlAcDbLine> allRectangleLines = new HashSet<crawlAcDbLine>();
            foreach (Rectangle rectangle in rectangles)
            {
                allRectangleLines.Add(rectangle.line1);
                allRectangleLines.Add(rectangle.line2);
                allRectangleLines.Add(rectangle.line3);
                allRectangleLines.Add(rectangle.line4);
            }


            foreach (crawlAcDbLine cLine in lines)
            {
                string rotated = "Rotated";
                if (cLine.StartPoint.X == cLine.EndPoint.X)
                    rotated = "Vertical";
                if (cLine.StartPoint.Y == cLine.EndPoint.Y)
                    rotated = "Horizontal";

                bool formsRectangle = allRectangleLines.Contains(cLine);

                sw.WriteLine(rotated + "; " + cLine.Length + "; " + formsRectangle);
            }

            sw.Close();
        }


        static void ObjJsons()
        {
            SqlDb sqlDB = new SqlDb();

            List<string> jsonObjs = sqlDB.GetObjectJsonByClassName("");

            string fileName = @"c:\Data\all.json";

            //https://msdn.microsoft.com/en-us/library/3aadshsx(v=vs.110).aspx
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileName, FileMode.CreateNew);
                using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
                {
                    writer.WriteLine("{\"objects\":[");

                    foreach (string jsonObj in jsonObjs)
                        if (jsonObj != "")
                        {
                            writer.WriteLine(jsonObj);
                            writer.Write(",");
                        }
                    writer.WriteLine("]}");
                }
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }
        }

        static void TextData()
        {
            SqlDb sqlDB = new SqlDb();
            string className = "AcDbText";

            List<string> jsonObjs = sqlDB.GetObjectJsonByClassName(className);

            string fileName = @"c:\Data\" + className + ".csv";

            //https://msdn.microsoft.com/en-us/library/3aadshsx(v=vs.110).aspx
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileName, FileMode.CreateNew);
                using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
                {
                    writer.WriteLine("Data" + "; " + "Value");

                    foreach (string jsonObj in jsonObjs)
                    {
                        crawlAcDbText cObject = jsonHelper.From<crawlAcDbText>(jsonObj);

                        writer.WriteLine(cObject.Position.ToString() + "; " + cObject.TextString);

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
            SqlDb sqlDB = new SqlDb();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            StreamWriter sw = new StreamWriter(@"F:\Data\LineData.csv");
            sw.WriteLine("Alignment" + "; " + "Length");

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                string rotated = "Rotated";
                if (cLine.StartPoint.X == cLine.EndPoint.X)
                    rotated = "Vertical";
                if (cLine.StartPoint.Y == cLine.EndPoint.Y)
                    rotated = "Horizontal";

                sw.WriteLine(rotated + "; " + cLine.Length);

            }

            sw.Close();
        }
    }
}
