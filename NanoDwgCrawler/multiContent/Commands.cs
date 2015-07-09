using System;
using Teigha.Runtime;
using HostMgd.ApplicationServices;
using System.IO;
using System.Windows.Forms;
using System.Data.SqlServerCe;
using Teigha.DatabaseServices;
using Teigha.Geometry;

using System.Diagnostics;

// Не работает вместе c Multicad.Runtime
using HostMgd.EditorInput;
using Platform = HostMgd;
using PlatformDb = Teigha;

//Использование определенных типов, которые определены и в платформе и в мультикаде
using Hatch = Teigha.DatabaseServices.Hatch;
using Point3d = Teigha.Geometry.Point3d;
using Polyline3d = Teigha.DatabaseServices.Polyline3d;

using TeighaApp = HostMgd.ApplicationServices.Application;
using System.Collections.Generic;
using RTree;


public class App : IExtensionApplication
{//http://through-the-interface.typepad.com/through_the_interface/2013/01/displaying-a-dialog-on-autocad-startup-using-net.html
    //Класс автоматически подгружает логер событий при загрузке приложения
    private int initCount = 0;
    public void Initialize()
    {
        if (initCount == 0)
        {
            // Crawl.Commands.Crawl();
            // Crawl.Commands.CrawlDirectory(@"\\FILESERVER\home\#АРХИВ 2014");
            initCount++;
        }
    }


    public void Terminate()
    {
    }
}

namespace Crawl
{

    public partial class Commands
    {
        Database acCurDb = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument.Database;
        Document acCurDoc = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument;
        Editor ed = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument.Editor;

        [CommandMethod("ogpTestDWGcrawl")]
        public static void ogpTestDWGcrawl()
        {
            //Показать диалог открытия папки
            //            FolderBrowserDialog fd = new FolderBrowserDialog();
            //            fd.RootFolder = Environment.SpecialFolder.Desktop;

            // if (fd.ShowDialog() == DialogResult.OK)
            Crawl(false);

        }

        public static void Crawl(bool closeAfterComplete = true)
        {
            SqlDb sqlDB = new SqlDb();
            //While Get random dwg from database that not scanned
            CrawlDocument crawlDoc = sqlDB.GetNewRandomUnscannedDocument();
            while (crawlDoc != null)
            {
                crawlAcDbDocument cDoc = new crawlAcDbDocument(crawlDoc);
                cDoc.sqlDB = sqlDB;
                cDoc.ScanDocument();
                crawlDoc = sqlDB.GetNewRandomUnscannedDocument();
            }
            if (closeAfterComplete)
                HostMgd.ApplicationServices.Application.Quit();
        }

        [CommandMethod("Clusters")]
        public static void Clusters()
        {
            SqlDb sqlDB = new SqlDb();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            RectangleTree rt = new RectangleTree();
            List<Rectangle> rectangles = new List<Rectangle>();

            int i = 0;
            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                Rectangle rec = new Rectangle(cLine.StartPoint, cLine.EndPoint);
                rectangles.Add(rec);

                rt.Add(rec);
                i++;
            }

            List<RectangleIntersection> intersections = new List<RectangleIntersection>();

            foreach (Rectangle rec in rectangles)
            {
                foreach (Rectangle recInt in rt.Intersections(rec))
                {


                }
            }


            var ri = new Rectangle(0,0,0,0);
            
                DrawRectangle(ri.MinPoint.X, ri.MinPoint.Y, ri.MaxPoint.X, ri.MaxPoint.Y);
            
        }

        [CommandMethod("ogpTestTree")]
        public static void ogpTestTree()
        {
            SqlDb sqlDB = new SqlDb();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            List<crawlAcDbLine> lines = new List<crawlAcDbLine>();

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                lines.Add(cLine);
            }

            RTree<string> tree = new RTree<string>(4, 2);

            int i = 0;
            foreach (crawlAcDbLine line in lines)
            {

                float x1 = Convert.ToSingle(line.StartPoint.X);
                float y1 = Convert.ToSingle(line.StartPoint.Y);
                float z1 = Convert.ToSingle(line.StartPoint.Z);
                float x2 = Convert.ToSingle(line.EndPoint.X);
                float y2 = Convert.ToSingle(line.EndPoint.Y);
                float z2 = Convert.ToSingle(line.EndPoint.Z);

                RTree.Rectangle rect = new RTree.Rectangle(x1, y1, x2, y2, z1, z2);

                tree.Add(rect, i.ToString());
                i++;
            }

            for (i = 0; i < tree.Count; i++)
            {
                var node = tree.getNode(i);

                if (!node.isLeaf())//&& node.getLevel() > 1 && node.getLevel() < tree.treeHeight)
                {
                    RTree.Rectangle rec = node.getMBR();
                    DrawRectangle(rec.min[0], rec.min[1], rec.max[0], rec.max[1], node.getLevel().ToString());
                }
            }

        }

        [CommandMethod("DrawRectangletest")]
        public static void DrawRectangletest()
        {
            DrawRectangle(0, 0, 1000, 1000);
        }

        private static void DrawRectangle(double x1, double y1, double x2, double y2, string text = "")
        {
            Document doc = TeighaApp.DocumentManager.MdiActiveDocument;
            Database acCurDb = doc.Database;
            using (Transaction tr = doc.Database.TransactionManager.StartTransaction())
            {
                // Open the Block table for read
                BlockTable acBlkTbl;
                acBlkTbl = tr.GetObject(
                    acCurDb.BlockTableId,
                    OpenMode.ForRead) as BlockTable;

                // Open the Block table record Model space for write
                BlockTableRecord acBlkTblRec;
                acBlkTblRec = tr.GetObject(
                    acBlkTbl[BlockTableRecord.ModelSpace],
                    OpenMode.ForWrite) as BlockTableRecord;

                // Create a copy of the object 
                Polyline pl = new Polyline();
                pl.SetDatabaseDefaults();

                pl.AddVertexAt(0, new Point2d(x1, y1), 0, 0, 0);
                pl.AddVertexAt(0, new Point2d(x1, y2), 0, 0, 0);
                pl.AddVertexAt(0, new Point2d(x2, y2), 0, 0, 0);
                pl.AddVertexAt(0, new Point2d(x2, y1), 0, 0, 0);
                pl.Closed = true;

                pl.Color = PlatformDb.Colors.Color.FromRgb(255, 0, 0);

                DBText txt = new DBText();
                txt.Color = PlatformDb.Colors.Color.FromRgb(255, 0, 0);
                txt.TextString = text;
                txt.Position = new Point3d(x1, y1, 0);
                txt.SetDatabaseDefaults();
                txt.Height = 250;

                acBlkTblRec.AppendEntity(txt);
                tr.AddNewlyCreatedDBObject(txt, true);

                // Add the cloned object to db
                acBlkTblRec.AppendEntity(pl);
                tr.AddNewlyCreatedDBObject(pl, true);

                tr.Commit();
            }

        }
    }
}
