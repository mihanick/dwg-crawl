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

namespace DwgDump
{
	public class Main : IExtensionApplication
	{
		// http://through-the-interface.typepad.com/through_the_interface/2013/01/displaying-a-dialog-on-autocad-startup-using-net.html
		// Класс автоматически подгружает логер событий при загрузке приложения
		private int initCount = 0;
		public void Initialize()
		{
			if (initCount == 0)
			{
				initCount++;
			}
			if (initCount == 1)
				Commands.WriteDocuments(true);
		}

		public void Terminate()
		{
		}
	}

	public class Commands
	{
		[CommandMethod("Dump2db")]
		public static void Dump2db()
		{
			WriteDocuments(false);
		}

		public static void WriteDocuments(bool closeAfterComplete = true)
		{
			DbMongo db = new DbMongo();
			//While Get random dwg from database that not scanned
			CrawlDocument crawlDoc = db.GetNewRandomUnscannedDocument();
			while (crawlDoc != null)
			{
				CrawlAcDbDocument cDoc = new CrawlAcDbDocument(crawlDoc)
				{
					db = db
				};
				cDoc.DumpDocument();
				crawlDoc = db.GetNewRandomUnscannedDocument();
			}
			if (closeAfterComplete)
				HostMgd.ApplicationServices.Application.Quit();
		}
	}
}
