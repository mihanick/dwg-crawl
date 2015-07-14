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
            DbMongo sqlDB = new DbMongo();
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
            // SqlDb sqlDB = new SqlDb(@"c:\Data\rectangle.sdf");
            SqlDb sqlDB = new SqlDb(@"c:\Data\SingleFile.sdf");
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            List<Rectangle> rectangles = new List<Rectangle>();

            int i = 0;
            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                if (cLine.Length > 0)
                {
                    Rectangle rec = new Rectangle(cLine.StartPoint, cLine.EndPoint);
                    rectangles.Add(rec);
                    DrawLine(cLine.StartPoint.X, cLine.StartPoint.Y, cLine.EndPoint.X, cLine.EndPoint.Y);
                }
                i++;
            }

            ClusterTree ct = new ClusterTree(rectangles.ToArray());

            foreach (ClusterTree.Cluster cluster in ct.Clusters)
            {
                if (cluster.Count > 2)
                    DrawRectangle(cluster.BoundBox.MinPoint.X, cluster.BoundBox.MinPoint.Y, cluster.BoundBox.MaxPoint.X, cluster.BoundBox.MaxPoint.Y);
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

        private static void DrawLine(double x1, double y1, double x2, double y2, string text = "")
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
                pl.AddVertexAt(0, new Point2d(x2, y2), 0, 0, 0);

                pl.Color = PlatformDb.Colors.Color.FromRgb(0, 255, 0);

                DBText txt = new DBText();
                txt.Color = PlatformDb.Colors.Color.FromRgb(0, 255, 0);
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
