using DwgDump.Db;
using HostMgd.ApplicationServices;
// Не работает вместе c Multicad.Runtime
using HostMgd.EditorInput;
using Teigha.DatabaseServices;
using Teigha.Geometry;
using Teigha.Runtime;
using Platform = HostMgd;
using PlatformDb = Teigha;
//Использование определенных типов, которые определены и в платформе и в мультикаде
using Point3d = Teigha.Geometry.Point3d;
using TeighaApp = HostMgd.ApplicationServices.Application;

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

namespace DwgDump

{
	public partial class Commands
    {
        Database acCurDb = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument.Database;
        Document acCurDoc = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument;
        Editor ed = Platform.ApplicationServices.Application.DocumentManager.MdiActiveDocument.Editor;

        [CommandMethod("Dump2db")]
        public static void Dump2db()
        {
            WriteDocuments(false);
        }

        public static void WriteDocuments(bool closeAfterComplete = true)
        {
            DbMongo db = new DbMongo("SingleFile");
            //While Get random dwg from database that not scanned
            CrawlDocument crawlDoc = db.GetNewRandomUnscannedDocument();
            while (crawlDoc != null)
            {
                crawlAcDbDocument cDoc = new crawlAcDbDocument(crawlDoc);
                cDoc.sqlDB = db;
                cDoc.ScanDocument();
                crawlDoc = db.GetNewRandomUnscannedDocument();
            }
            if (closeAfterComplete)
                HostMgd.ApplicationServices.Application.Quit();
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
