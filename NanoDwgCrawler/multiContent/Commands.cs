using System;
using Teigha.Runtime;
using HostMgd.ApplicationServices;
using System.IO;
using System.Windows.Forms;
using System.Data.SqlServerCe;
using Teigha.DatabaseServices;
using Teigha.Geometry;

// Не работает вместе c Multicad.Runtime
using HostMgd.EditorInput;
using Platform = HostMgd;
using PlatformDb = Teigha;

//Использование определенных типов, которые определены и в платформе и в мультикаде
using Hatch = Teigha.DatabaseServices.Hatch;
using Point3d = Teigha.Geometry.Point3d;
using Polyline3d = Teigha.DatabaseServices.Polyline3d;

using TeighaApp = HostMgd.ApplicationServices.Application;


public class App : IExtensionApplication
{//http://through-the-interface.typepad.com/through_the_interface/2013/01/displaying-a-dialog-on-autocad-startup-using-net.html
    //Класс автоматически подгружает логер событий при загрузке приложения
    private int initCount = 0;
    public void Initialize()
    {
        if (initCount == 0)
        {
            Crawl.Commands.Crawl();
            //Crawl.Commands.CrawlDirectory(@"\\FILESERVER\home\#АРХИВ 2014");
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
            FolderBrowserDialog fd = new FolderBrowserDialog();
            fd.RootFolder = Environment.SpecialFolder.Desktop;

            if (fd.ShowDialog() == DialogResult.OK)
            {
                Crawl();
            }
        }

        public static void Crawl()
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
            HostMgd.ApplicationServices.Application.Quit();
        }
    }
}
